# DINOv2 Register Exploration

This repository contains code and resources for exploring register tokens in the DINOv2 model, specifically using the ImageNet100 dataset.

## Datasets

This project uses the following datasets:

### 1. ImageNet100 Dataset
[Download the ImageNet100 Dataset](https://www.kaggle.com/datasets/ambityga/imagenet100/data) and save it in the folder named `Imagenet100`.

### 2. COCO Dataset
For this project, we used the `COCO2014 Dataset` from Hugging Face.
```
from datasets import load_dataset
ds = load_dataset("detection-datasets/coco")
```
We also used the COCO 2017 Dataset from the following website: 
https://cocodataset.org/#download


## Precomputed Embeddings

Precomputed embeddings for the ImageNet100 dataset are provided in this repository in the following directories:
- `Imagenet100/train_embeddings`
- `Imagenet100/val_embeddings`

You can use these embeddings directly without recomputing.

### Recomputing the Embeddings

If you want to recompute the embeddings for validation or testing purposes, you can run the following script:

```bash
python compute_embeddings.py


