from transformers import AutoModel,AutoTokenizer,pipeline
from datasets import load_dataset,Dataset

# HF_ENDPOINT=https://hf-mirror.com python test04.py 

tokenizer = AutoTokenizer.from_pretrained("./tokenizer/roberta")

# dataset = load_dataset("kuroneko5943/stock11","finance")
# dataset = load_dataset("kuroneko5943/stock11","finance",split="train")
# dataset = load_dataset("kuroneko5943/stock11","finance",split=["train[:6000]","train[6000:]"])
# dataset = load_dataset("kuroneko5943/stock11","finance",split={"train":"train[:6000]","test":"train[6000:]"})
dataset = load_dataset("kuroneko5943/stock11","finance",split={"train":"train[:80%]","test":"train[80%:]"})
# print(dataset)

def handle(data):
    # print(data)
    input = tokenizer(data["sentence"])
    input["labels"] = data["label"]
    # print(input)
    return input

# train_dataset = dataset["train"].map(handle)
train_dataset = dataset["train"].map(handle,remove_columns=dataset["train"].column_names)
print(train_dataset)
# print(dataset["train"][0])
