from transformers import AutoModel,AutoTokenizer,pipeline,AutoModelForSequenceClassification
from datasets import load_dataset,Dataset
import torch

# HF_ENDPOINT=https://hf-mirror.com python test04.py 

# model = AutoModel.from_pretrained("hfl/rbt3")
model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
# model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3",num_labels=3)
model.classifier = torch.nn.Linear(768,2,bias=False)
# print(model)
print(model.config)

tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")

strs = ["我爱北京天安门,天安门上太阳升！","今天天气真不错！"]
tokens = tokenizer(strs,padding="longest",return_tensors="pt")
tokens["labels"] = torch.tensor([1,0])
print(model(**tokens))
# print(model(**tokens,output_attentions=True))