from transformers import AutoModel,AutoTokenizer,pipeline,AutoModelForSequenceClassification
from datasets import load_dataset,Dataset
import torch
import evaluate

# HF_ENDPOINT=https://hf-mirror.com python test04.py 

accuracy = evaluate.load("accuracy")
# accuracy = evaluate.load("f1")
print(accuracy.description)

# acc = accuracy.compute(references=[0,1,0,1],predictions=[1,0,0,1])

# for ref,pred in zip([0,1,0,1], [1,0,0,1]):
#     accuracy.add(references=ref,predictions=pred)
# acc = accuracy.compute()

# for ref,pred in zip([[0,1],[0,1]], [[1,0],[0,1]]):
#     accuracy.add_batch(references=ref,predictions=pred)
# acc = accuracy.compute()

# print(acc)

clf_metrics = evaluate.combine(["accuracy","f1","precision","recall"])
acc = clf_metrics.compute(predictions=[0,1,0],references=[0,1,1])
print(acc)