"""
run:
    HF_ENDPOINT=https://hf-mirror.com python Lora.py
"""

import os 
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

from transformers import (AutoModelForSequenceClassification, # 分类任务 
                          AutoTokenizer,
                          BitsAndBytesConfig 
                          )

from peft import (LoraConfig,
                  get_peft_model,
                  TaskType,
                  )

from datasets import load_dataset,load_from_disk,DownloadConfig
import torch
from trl import RewardConfig,RewardTrainer

_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
_tokenizer.add_special_tokens({"bos_token": _tokenizer.eos_token,
                               "pad_token": _tokenizer.eos_token})
_tokenizer.bos_token_id = _tokenizer.eos_token_id
_tokenizer.pad_token_id = _tokenizer.eos_token_id

# 主网络nf4量化
_bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_use_double_quant=True, #开启双量化
                                 bnb_4bit_quant_type="nf4",      # nf4量化
                                 bnb_4bit_compute_dtype=torch.float32  # float16 bfloat16 不炸就可以
                                 )

_model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen2-0.5B-Instruct",
                                              low_cpu_mem_usage=True,
                                              quantization_config=_bnb_config)

# _dataset = load_dataset("json",data_files="./data/alpaca_gpt4_data_zh.json",split="train")
# _dataset = load_dataset("parquet",data_files="./data/train-00000-of-00001.parquet",split="train")
# _dataset = load_dataset("Dahoas/rm-static",split="train")
# _dataset = load_dataset("wenbopan/Chinese-dpo-pairs",split="train")
_data_files = {"train":"train-00000-of-00001.parquet"} #,"test":"test-00000-of-00001-8c7c51afc6d45980.parquet"}
_dataset = load_dataset("parquet", data_dir='./wenbopan/train', split="train")
print(_dataset[0])
# datasets_config = DownloadConfig(resume_download=True, max_retries=100) 
# _dataset = load_dataset( "wenbopan/Chinese-dpo-pairs", download_config=datasets_config,cache_dir="./data_f_hub", split="train" )


def preprocess_dataset(data,tokenizer=_tokenizer):
    chosen = tokenizer(f"<|im_start|>user\n{data['prompt']}<|im_end|><|im_start|>assistant\n{data['chosen']}<|im_end|>")
    rejected = tokenizer(f"<|im_start|>user\n{data['prompt']}<|im_end|><|im_start|>assistant\n{data['rejected']}<|im_end|>")
    return{
        "input_ids_chosen": chosen["input_ids"],
        "attention_mask_chosen": chosen["attention_mask"],
        "input_ids_rejected": rejected["input_ids"],
        "attention_mask_rejected":rejected["attention_mask"]
    }


_dataset = _dataset.map(preprocess_dataset,
                        batch_size=True,
                        num_proc=10,
                        remove_columns=_dataset.column_names)
_dataset = _dataset.shuffle()


config = LoraConfig(task_type=TaskType.SEQ_CLS,
                    # target_modules=["q_proj","k_proj","v_proj","up_proj","down_proj","gate_proj"],
                    target_modules="all-linear",
                    modules_to_save=["score"]   # 
                    )

_model = get_peft_model(_model,config)
_model.config.pad_token_id = _model.config.eos_token_id

# _model.print_trainable_parameters()
_model.enable_input_require_grads()

_training_args= RewardConfig(
    output_dir="checkpoints/Reward",
    # dataset_text_field="text",     # 指定数据格式
    # max_seq_length=512,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,  # 16G显存可以训练7b模型
    # logging_steps=10,
    num_train_epochs=6,
    save_steps=100,
    save_total_limit=3,
    optim="paged_adamw_32bit" # 分页优化器
)
 
_trainer = RewardTrainer(
    model=_model,
    tokenizer=_tokenizer,
    args=_training_args,
    train_dataset=_dataset
)

_trainer.train()