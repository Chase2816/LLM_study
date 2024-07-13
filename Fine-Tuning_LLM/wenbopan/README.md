---
license: mit
dataset_info:
  config_name: train
  features:
  - name: prompt
    dtype: string
  - name: system
    dtype: string
  - name: chosen
    dtype: string
  - name: rejected
    dtype: string
  - name: source
    dtype: string
  - name: id
    dtype: string
  splits:
  - name: train
    num_bytes: 28322152
    num_examples: 10735
  download_size: 17430997
  dataset_size: 28322152
configs:
- config_name: train
  data_files:
  - split: train
    path: train/train-*
  default: true
language:
- zh
---

# Dataset Card for Chinese-dpo-pairs

Well-curated 10K reference pairs in Chinese. Data are created by GPT-3.5 translation from multiple sources, including:

- flan_v2, sharegpt, ultrachat, evol_instruct and false_qa. Sampled from [argilla/ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)
- open_orca. From [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs)
- truthy_dpo. From [jondurbin/truthy-dpo-v0.1](https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1)

To ensure quality, I originally translated over 30K samples, then dropped all tranlations with unmatched line number or topic. The dataset is best used together with above English dataset.