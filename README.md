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
```

## Notebooks

Please find the working code of this project in the `notebooks` folder. 
If you are checking this project the files should be read in the following order:

- **AttentionMaps**: Contains the functions used to extract the attention maps from DINOv2.  
  Note: To properly extract the attention maps, we had to modify some original DINOv2 functions (the modified version is attached in this repository under `dinov2`). Running the function with the original DINOv2 code will not work.

- **ImageNet100**: Register exploration and classification models on the ImageNet100 (Image Classification) dataset.

- **COCO2014**: Register exploration and classification models on the COCO 2014 (Object Detection) dataset.

- **COCO2017**: Register exploration and classification models on the COCO 2017 (Object Detection) dataset.


