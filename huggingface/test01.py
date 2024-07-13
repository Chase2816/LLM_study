# HF_ENDPOINT=https://hf-mirror.com python main.py

import gradio as gr
from transformers import *

pipe = pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese")
# print(pipe("oh,my god!"))

gr.Interface.from_pipeline(pipe).launch(share=True,server_name="0.0.0.0",server_port=60012)

# from transformers.pipelines.question_answering import QuestionAnsweringPipeline


# huggingface-cli download --resume-download xiaolxl/GuoFeng4_XL 
