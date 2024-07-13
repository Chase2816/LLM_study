import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          TrainingArguments,
                          Trainer,
                          DataCollatorWithPadding
                          )
from datasets import load_dataset
import evaluate


_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-large")
_model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-macbert-large")

# 第四种显存优化 模型加载后，找出bert的参数冻结 14g-->2g
for name,param in _model.bert.named_parameters():
    print(name,param.dtype)
    param.requires_grad=False
    
_dataset = load_dataset("csv",data_files="ChnSentiCorp_htl_all.csv",split="train")

_dataset = _dataset.filter(lambda x:x["review"] is not None)

# 第五种显存优化 修改输入数据长度  max_length
def preprocess_dataset(data,tokenizer=_tokenizer):
    _rst = tokenizer(data["review"],max_length=512,truncation=True,padding="max_length")
    _rst["labels"] = data["label"]
    return _rst


_dataset = _dataset.map(preprocess_dataset,remove_columns=_dataset.column_names)
_dataset = _dataset.shuffle()
_datasets = _dataset.train_test_split(test_size=0.2)

"""
显存优化：
1. FP32改为混合精度运算FP16、BF16、INTx
2. 梯度累加
3. 更换优化器
4. 冻结部分层
5. 修改输入数据长度
6. 使用gradient checkpointing
7. 其它优化方法:
    Zero0/1/2/3、offloader、FlashAttention、QLora
"""
training_args= TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    # gradient_accumulation_steps=1, #12 #第二种显存优化，梯度累加 14g-->6.3g
    # gradient_checkpointing=True,  # 第六种显存优化 梯度检查点，没有参数不保存梯度，计算量换显存 14g-->6g
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=2e-5,
    weight_decay=0.1,
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    logging_steps=10,
    # fp16=True, # 第一种显存优化 14g--->11g
    # optim="adafactor" #第三种显存优化 , 更换优化器 sgd:14g-->11g adafactor:14g-->
)


_accuracy_model=evaluate.load("accuracy")

def compute_accuracy(result):
    _predictions, _labels = result
    _predictions = _predictions.argmax(-1)
    _accuracy = _accuracy_model(predictions=_predictions,labels=_labels)
    return _accuracy


trainer = Trainer(
    model=_model,
    tokenizer=_tokenizer,
    args=training_args,
    train_dataset=_datasets["train"],
    eval_dataset=_datasets["test"],
    data_collator=DataCollatorWithPadding(tokenizer=_tokenizer,padding=True),
    compute_metrics=compute_accuracy
)

trainer.train()
# trainer.evaluate()
# print(trainer.predict(test_dataset))


