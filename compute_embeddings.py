import os
import torch
import json
from dataloader import load_imagenet_data
from model_utils import initialize_model
from embedding_utils import compute_embeddings, save_embeddings

# Set directories and parameters
cwd = os.getcwd()
ROOT_DIR = os.path.join(cwd, "Imagenet100")
val_folder = "val.X"
train_folders = ["train.X1", "train.X2", "train.X3", "train.X4"]
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Load class labels
labels_file = os.path.join(ROOT_DIR, "Labels.json")
with open(labels_file, 'r') as f:
    class_labels = json.load(f)

# Load train and validation data
df_train, df_val = load_imagenet_data(ROOT_DIR, class_labels, val_folder, train_folders)

# Initialize the model
dinov2_vits14 = initialize_model(device)

# Compute and save embeddings for train data
train_embeddings, excluded_train = compute_embeddings(list(df_train['path']), dinov2_vits14, device)
save_embeddings(train_embeddings, os.path.join(ROOT_DIR, 'train_embeddings'))

# Compute and save embeddings for validation data
val_embeddings, excluded_val = compute_embeddings(list(df_val['path']), dinov2_vits14, device)
save_embeddings(val_embeddings, os.path.join(ROOT_DIR, 'val_embeddings'))

print("Embeddings computation completed. Check the Imagenet100 folder for saved files.")
