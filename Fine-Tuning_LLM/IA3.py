"""
IA3的思想:
    抑制和放大内部激活,通过可学习的向量对激活值进行抑制或放大。
    具体来说,会对K、VFFN三部分的值进行调整,训练过程中同样冻结原始模型的权重,
    只更新可学习的部分向量部分。训练完成后,与Lora类似,也可以将学习部分的参数与原始权重合并,没有额外推理开销。
run:
    HF_ENDPOINT=https://hf-mirror.com python Lora.py
"""

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForSeq2Seq
                          )

from peft import (IA3Config,
                  get_peft_model,
                  TaskType,
                  )

from datasets import load_dataset,load_from_disk,Dataset
import evaluate


# _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
# _model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
_tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")
_model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")

# _name找target_modules=["q_proj","k_proj","v_proj","up_proj","down_proj","gate_proj"],
# for _name,_param in _model.named_parameters():
#     print(_name)
# exit()

_dataset = load_dataset("json",data_files="./data/alpaca_gpt4_data_zh_3147.json",split="train")
# _dataset = load_dataset("json",data_files="shibing624/alpaca-zh.json",split="train")
# _dataset = Dataset.from_json("./data/alpaca_gpt4_data_zh.json")

def preprocess_dataset(example):
    MAX_LENGTH = 256
    _input_ids,_attention_mask,_labels = [],[],[]
    _instruction = _tokenizer("\n".join(["Human: ",example["instruction"]]).strip() + "\n\nAssistant: ")
    _response = _tokenizer(example["output"] + _tokenizer.eos_token)
    _input_ids = _instruction["input_ids"] + _response["input_ids"]
    _attention_mask = _instruction["attention_mask"] + _response["attention_mask"]
    _labels = [-100] * len(_instruction["input_ids"]) + _response["input_ids"]
    if len(_input_ids) > MAX_LENGTH:
        _input_ids = _input_ids[:MAX_LENGTH]
        _attention_mask = _attention_mask[:MAX_LENGTH]
        _labels = _labels[:MAX_LENGTH]
    return {
        "input ids": _input_ids,
        "attention": _attention_mask,
        "labels": _labels
    }

def preprocess_dataset2(example):
    MAX_LENGTH = 256
    _instruction = _tokenizer(f"Human: {example['instruction']}\n\nAssistant: ")
    _response = _tokenizer(example['output'] + _tokenizer.eos_token)
    _input_ids = _instruction['input_ids'] + _response['input_ids']
    _attention_mask = _instruction['attention_mask'] + _response['attention_mask']
    _labels = [-100] * len(_instruction['input_ids']) + _response['input_ids']
    if len(_input_ids) > MAX_LENGTH:
        _input_ids = _input_ids[:MAX_LENGTH]
        _attention_mask = _attention_mask[:MAX_LENGTH]
        _labels = _labels[:MAX_LENGTH]
    return {"input_ids": _input_ids, "attention_mask": _attention_mask, "labels": _labels}

_dataset = _dataset.map(preprocess_dataset2,remove_columns=_dataset.column_names)
# _dataset = _dataset.map(preprocess_dataset,remove_columns=_dataset.column_names)
_dataset = _dataset.shuffle()

# 硬模式
config = IA3Config(task_type=TaskType.CAUSAL_LM)

_model = get_peft_model(_model,config)

# _model.print_trainable_parameters()

_training_args= TrainingArguments(
    output_dir="checkpoints/IA3",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=20,
    save_total_limit=3
)
 
trainer = Trainer(
    model=_model,
    args=_training_args,
    train_dataset=_dataset,   
    data_collator=DataCollatorForSeq2Seq(tokenizer=_tokenizer,padding=True),
)

# trainer.train()

# 模型推理

from transformers import pipeline
from peft import PeftModel

peft_model = PeftModel.from_pretrained(model=_model,model_id="./checkpoints/IA3/checkpoint-602")

peft_model = peft_model.cuda()
ipt = _tokenizer("Human: {}\n{}".format("在服务行业中如何使用人工智能？","").strip() + "\n\nAssistant: ",return_tensors="pt").to(peft_model.device)
print(_tokenizer.decode(peft_model.generate(**ipt,max_length=128,do_sample=True)[0],skip_special_tokens=True))
