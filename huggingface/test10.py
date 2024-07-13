from transformers import AutoModel,AutoTokenizer,pipeline,AutoModelForSequenceClassification
from datasets import load_dataset,Dataset
import torch
import evaluate
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments,Trainer
import numpy as np 

"""
run:
    HF_ENDPOINT=https://hf-mirror.com python test04.py 
tensorboard:
    tensorboard --logdir=test_trainer/runs --port 60014 --bind_all
"""
tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3",force_download=False)
# model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3",num_labels=3,force_download=False)


def handle(data):
    # print(data)
    v = tokenizer(data["sentence"])
    v["labels"] = data["label"]
    # print(v)
    return v

dataset = load_dataset("kuroneko5943/stock11","finance")
# print(dataset)
train_dataset = dataset["train"].map(handle,remove_columns=dataset["train"].column_names)
validation_dataset = dataset["validation"].map(handle,remove_columns=dataset["validation"].column_names)
test_dataset = dataset["test"].map(handle,remove_columns=dataset["test"].column_names)
# print(train_dataset)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred,metric=metric):
    logits,labels = eval_pred
    predictions = np.argmax(logits,axis=-1)
    return metric.compute(predictions=predictions,references=labels)

training_args = TrainingArguments(output_dir="test_trainer",
                                #   num_train_epochs=50,
                                  per_device_train_batch_size=8,
                                  per_device_eval_batch_size=4,
                                  logging_steps=1000)
# print(training_args)

trainer = Trainer(
    tokenizer=tokenizer,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer,padding=True)
)

print(trainer.train())
print(trainer.evaluate())
print(trainer.predict(test_dataset))
