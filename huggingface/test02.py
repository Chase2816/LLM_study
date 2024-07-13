from transformers import AutoModel,AutoTokenizer,pipeline

model = AutoModel.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")

pp = pipeline(task="text-classification",model=model,tokenizer=tokenizer)
print("hello world!")