import os
import pandas as pd
import json

def load_imagenet_data(root_dir, class_labels, val_folder, train_folders):
    """
    Load ImageNet100 training and validation data.
    
    Args:
        root_dir (str): Root directory of the ImageNet100 dataset.
        class_labels (dict): Dictionary with class labels.
        val_folder (str): Folder name for validation data.
        train_folders (list): List of folder names for training data.
    
    Returns:
        df_train (DataFrame): DataFrame containing training image paths and labels.
        df_val (DataFrame): DataFrame containing validation image paths and labels.
    """
    train_paths = [os.path.join(root_dir, folder) for folder in train_folders]
    val_path = os.path.join(root_dir, val_folder)

    # Load train data
    train_images = []
    for train_path in train_paths:
        for label in os.listdir(train_path):
            curr_folder = os.path.join(train_path, label)
            all_files = os.listdir(curr_folder)
            for filename in all_files[:100]:  # Limit to 100 files per class for speed
                train_images.append({
                    'path': os.path.join(curr_folder, filename),
                    'label': class_labels[label],
                })
    df_train = pd.DataFrame(train_images)

    # Load validation data
    val_images = []
    for label in os.listdir(val_path):
        curr_folder = os.path.join(val_path, label)
        for filename in os.listdir(curr_folder):
            val_images.append({
                'path': os.path.join(curr_folder, filename),
                'label': class_labels[label],
            })
    df_val = pd.DataFrame(val_images)

    return df_train, df_val


def load_labels_mapping(root_dir: str, labels_filename: str) -> dict:
    """
    Load the labels mapping from the specified directory.
    
    Args:
        root_dir (str): Root directory where the dataset is stored.
        labels_filename (str): Name of the JSON file containing the labels mapping.
    
    Returns:
        dict: A dictionary with labels mapping.
    """
    labels_file = os.path.join(root_dir, labels_filename)
    with open(labels_file, 'r') as f:
        labels_mapping = json.load(f)
    return labels_mapping
