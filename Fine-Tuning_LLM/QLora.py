"""
Lora原理:
    训练时,输入分别与原始权重和两个低秩矩阵进行计算,共同得到最终结果,优化则仅优化A和B
    训练完成后,可以将两个低秩矩阵与原始模型中的权重进行合并,合并后的模型与原始模型无异,避免了推理期间Prompt系列方法带来的额外计算量
run:
    HF_ENDPOINT=https://hf-mirror.com python Lora.py
"""

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForSeq2Seq,
                          BitsAndBytesConfig 
                          )

from peft import (LoraConfig,
                  get_peft_model,
                  TaskType,
                  )

from datasets import load_dataset,load_from_disk,Dataset
import evaluate
import torch

_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# 主网络量化
_bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_use_double_quant=True, #开启双量化
                                 bnb_4bit_quant_type="nf4",      # nf4量化
                                 bnb_4bit_compute_dtype=torch.float32  # float16 bfloat16 不炸就可以
                                 )

_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct",
                                              low_cpu_mem_usage=True,
                                              quantization_config=_bnb_config)

_dataset = load_dataset("json",data_files="./data/alpaca_gpt4_data_zh.json",split="train")

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


config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                    # target_modules=["q_proj","k_proj","v_proj","up_proj","down_proj","gate_proj"],
                    target_modules="all-linear",
                    modules_to_save=["word_embaddings"]
                    )

_model = get_peft_model(_model,config)

# _model.print_trainable_parameters()
_model.enable_input_require_grads() # 不开启会报错没有路径反向传播     gradient_checkpointing=True,  # 16G显存可以训练7b模型

_training_args= TrainingArguments(
    output_dir="checkpoints/QLora",
    per_device_train_batch_size=10,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,  # 16G显存可以训练7b模型
    logging_steps=10,
    num_train_epochs=6,
    save_steps=100,
    save_total_limit=3,
    optim="paged_adamw_32bit" # 分页优化器
)
 
trainer = Trainer(
    model=_model,
    args=_training_args,
    train_dataset=_dataset,   
    data_collator=DataCollatorForSeq2Seq(tokenizer=_tokenizer,padding=True),
)

trainer.train()