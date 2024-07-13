import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForSeq2Seq
                          )

from peft import (PromptEncoderConfig,
                  get_peft_model,
                  TaskType,
                  PromptEncoderReparameterizationType)

from datasets import load_dataset,load_from_disk,Dataset
import evaluate


"""
P-Tuning的思想: 
    在Prompt-Tuning的基础上,
    对Prompt部分进行进一步的编码计算,加速收敛. 
    具体来说,PEFT中支持两种编码方式,一种是LSTM,一种是MLP. 
    与Prompt-Tuning不同的是Prompt的形式只有Soft Prompt.
"""

# _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
# _model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
_tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")
_model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")

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
config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM,
                             num_virtual_tokens=10, # 越大越好
                             encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP, # MLP/LSTM
                             encoder_dropout=0.1,
                             encoder_num_layers=5,
                             encoder_hidden_size=1024
                             )

_model = get_peft_model(_model,config)

_training_args= TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=20
)
 
trainer = Trainer(
    model=_model,
    tokenizer=_tokenizer,
    args=_training_args,
    train_dataset=_dataset,   
    data_collator=DataCollatorForSeq2Seq(tokenizer=_tokenizer,padding=True),
)

trainer.train()


# 模型推理

from transformers import pipeline
from peft import PeftModel

peft_model = PeftModel.from_pretrained(model=_model,model_id="./checkpoints")

peft_model = peft_model.cuda()
ipt = _tokenizer("Human: {}\n{}".format("考试有哪些技巧？","").strip() + "\n\nAssistant: ",return_tensors="pt").to(peft_model.device)
print(_tokenizer.decode(peft_model.generate(**ipt,max_length=128,do_sample=True)[0],skip_special_tokens=True))
