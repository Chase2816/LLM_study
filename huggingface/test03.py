from transformers import AutoModel,AutoTokenizer,pipeline

# model = AutoModel.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
# tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
tokenizer = AutoTokenizer.from_pretrained("./tokenizer/roberta")
print(type(tokenizer))


# tokenizer.save_pretrained("./tookenuizer/roberta")

strs = ["我爱北京天安门,天安门上太阳升！","今天天气真不错！"]
# ids = tokenizer.encode(strs,add_special_tokens=False)
# print(ids)
# text = tokenizer.decode(ids)
# print(text)
# print(tokenizer(strs))
# print(tokenizer.tokenize(strs))

# tokens = tokenizer(strs,padding="longest")
# tokens = tokenizer(strs,padding="max_length",max_length=50)
# tokens = tokenizer(strs,padding="max_length",max_length=2,truncation=True)
tokens = tokenizer(strs,padding="max_length",max_length=2,truncation=True,return_tensors="pt")
print(tokens)