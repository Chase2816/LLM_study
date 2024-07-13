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
                          BitsAndBytesConfig,
                          pipeline
                          )

from peft import (LoraConfig,
                  get_peft_model,
                  TaskType,
                  )

from datasets import load_dataset,load_from_disk,DownloadConfig
import torch
from trl import PPOConfig,PPOTrainer,AutoModelForCausalLMWithValueHead

_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
_r_model = AutoModelForSequenceClassification.from_pretrained("./checkpoints/Reward/checkpoint-200")

_reward_model = pipeline(task="text-classification",model=_r_model,tokenizer=_tokenizer)


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

_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                    # target_modules=["q_proj","k_proj","v_proj","up_proj","down_proj","gate_proj"],
                    target_modules="all-linear",
                    )

# 强化学习核心算价值
_model = AutoModelForCausalLMWithValueHead.from_pretrained("Qwen/Qwen2-0.5B-Instruct",
                                                           low_cpu_mem_usage=True,
                                                           quantization_config=_bnb_config,
                                                           peft_config=_config,
                                                           )

_ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("Qwen/Qwen2-0.5B-Instruct",
                                                           low_cpu_mem_usage=True,
                                                           quantization_config=_bnb_config
                                                           )

_dataset = load_dataset("json",data_files="./data/alpaca_gpt4_data_zh_3147.json",split="train")

# _dataset = load_dataset("json",data_files="./data/alpaca_gpt4_data_zh.json",split="train")
# _dataset = load_dataset("parquet",data_files="./data/train-00000-of-00001.parquet",split="train")
# _dataset = load_dataset("Dahoas/rm-static",split="train")
# _dataset = load_dataset("wenbopan/Chinese-dpo-pairs",split="train")
# _data_files = {"train":"train-00000-of-00001.parquet"} #,"test":"test-00000-of-00001-8c7c51afc6d45980.parquet"}
# _dataset = load_dataset("parquet", data_dir='./wenbopan/train', split="train")
# print(_dataset[0])
# datasets_config = DownloadConfig(resume_download=True, max_retries=100) 
# _dataset = load_dataset( "wenbopan/Chinese-dpo-pairs", download_config=datasets_config,cache_dir="./data_f_hub", split="train" )


def preprocess_dataset(data,tokenizer=_tokenizer):
    # instruction = tokenizer(f"<im_start>system\n{example['instruction']}<|im_end|>\n<|im_start|>user{example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens

    question = tokenizer(f"<|im_start|>user\n{data['instruction']}<|im_end|><|im_start|>assistant\n{data['output']}<|im_end|>")
    return question
    # return {"input_ids":question["input_ids"],"attention_mask":question["attention_mask"]}

_dataset = _dataset.map(preprocess_dataset,
                        batch_size=True,
                        num_proc=10,
                        remove_columns=_dataset.column_names)
_dataset = _dataset.shuffle()




# _model = get_peft_model(_model,_config)
_model.config.pad_token_id = _model.config.eos_token_id

# _model.print_trainable_parameters()
# _model.enable_input_require_grads()

_training_args= PPOConfig(
    ppo_epochs=1
)
 
_ppo_trainer = PPOTrainer(
    model=_model,
    ref_model=_ref_model,
    tokenizer=_tokenizer,
    # args=_training_args,
    dataset=_dataset,
    config=_training_args    
)

generation_kwargs = {
    "min_Length":-1,
    "top_k":0.0,       # 选多少
    "top_p":0.8,       # 生成几句话
    "do_sample":True,
    "pad_token_id":_tokenizer.eos_token_id,
}

from tqdm import tqdm
epochs = 10
for epoch in tqdm(range(epochs),"epoch: "):
    for batch in tqdm(_ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]
        
        #### Get response from SFTModel
        response_tensors = _ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [_tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = _reward_model(texts)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
       
        ### Run PPO step
        stats = _ppo_trainer.step(query_tensors,response_tensors,rewards)
        _ppo_trainer.log_stats(stats,batch,rewards)
        
### Save model
_ppo_trainer.save_pretrained("checkpoints/PPO/my_ppo_model")