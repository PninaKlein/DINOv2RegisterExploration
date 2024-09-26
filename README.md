# DINOv2RegisterExploration
In this project we explore the content saved in the Register token embeddings retrieved from the DINOv2 Model 

# Datasets

This project uses the following datasets:

## 1. ImageNet100 Dataset
[Download the ImageNet100 Dataset](https://www.kaggle.com/datasets/ambityga/imagenet100/data) and save it in the folder named `Imagenet100`.

## 2. COCO Dataset
For this project, we used the `COCO2014 Dataset` from Hugging Face.
```
from datasets import load_dataset
ds = load_dataset("detection-datasets/coco")
```

