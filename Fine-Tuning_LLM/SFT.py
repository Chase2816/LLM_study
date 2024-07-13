"""
run:
    HF_ENDPOINT=https://hf-mirror.com python Lora.py
"""

import os 
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig 
                          )

from peft import (LoraConfig,
                  get_peft_model,
                  TaskType,
                  )

from datasets import load_dataset,load_from_disk,Dataset
import torch
from trl import SFTConfig,SFTTrainer

_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
_tokenizer.add_special_tokens({"bos_token": _tokenizer.eos_token})
_tokenizer.bos_token_id = _tokenizer.eos_token_id

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

def preprocess_dataset_sft(data):
    return{
        "text": f"<|im_start|>user\n{data['instruction']}<|im_end|><|im_start|>assistant\n{data['output']}<|im_end|>"
    }

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

_dataset = _dataset.map(preprocess_dataset_sft,
                        batch_size=True,
                        num_proc=10,
                        remove_columns=_dataset.column_names)
# _dataset = _dataset.map(preprocess_dataset2,remove_columns=_dataset.column_names)
# _dataset = _dataset.map(preprocess_dataset,remove_columns=_dataset.column_names)
_dataset = _dataset.shuffle()


config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                    # target_modules=["q_proj","k_proj","v_proj","up_proj","down_proj","gate_proj"],
                    target_modules="all-linear",
                    # modules_to_save=["word_embaddings"]
                    )

_model = get_peft_model(_model,config)

# _model.print_trainable_parameters()
_model.enable_input_require_grads()

_training_args= SFTConfig(
    output_dir="checkpoints/SFT",
    dataset_text_field="text",     # 指定数据格式
    max_seq_length=512,
    per_device_train_batch_size=10,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,  # 16G显存可以训练7b模型
    # logging_steps=10,
    num_train_epochs=6,
    save_steps=100,
    save_total_limit=3,
    optim="paged_adamw_32bit" # 分页优化器
)
 
_trainer = SFTTrainer(
    model=_model,
    tokenizer=_tokenizer,
    args=_training_args,
    train_dataset=_dataset
)

_trainer.train()