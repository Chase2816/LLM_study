import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          pipeline,
                          AutoModelForCausalLM
                          )
from datasets import load_dataset,Dataset
import evaluate


_tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")
_model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")

pp = pipeline(task="text-generation",tokenizer=_tokenizer,model=_model)
print(pp("Human: 爸爸再婚，我是不是就有了个新娘？\n\nAssistant:"))
