"""
Lora原理:
    训练时,输入分别与原始权重和两个低秩矩阵进行计算,共同得到最终结果,优化则仅优化A和B
    训练完成后,可以将两个低秩矩阵与原始模型中的权重进行合并,合并后的模型与原始模型无异,避免了推理期间Prompt系列方法带来的额外计算量
run:
    HF_ENDPOINT=https://hf-mirror.com python Lora.py
"""


# 模型推理

from transformers import (pipeline,
                          AutoTokenizer,
                          AutoModelForCausalLM
                        )
from peft import PeftModel

_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

peft_model = PeftModel.from_pretrained(model=_model,model_id="./checkpoints/Lora/checkpoint-1203")

# 合并为一个模型
# peft_model = peft_model.merge_and_unload()
# peft_model.sava_pretrained("myqwen2-0.5b")

# peft_model = peft_model.cuda()
pipe = pipeline("text-generation",model=_model,tokenizer=_tokenizer)
ipt = f"User: 你是谁？Assistant:"
print(pipe(ipt))