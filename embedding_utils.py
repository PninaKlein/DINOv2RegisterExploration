import torch
import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image

def compute_embeddings(files, model, device):
    """
    Compute embeddings for a list of images using the DINOv2 model.
    
    Args:
        files (list): List of image file paths.
        model: The DINOv2 model.
        device (torch.device): Device to run the model on (CPU or GPU).
    
    Returns:
        all_embeddings (dict): Dictionary with computed embeddings for each file.
        excluded_files (list): List of files that failed to process.
    """
    all_embeddings = {}
    excluded_files = []
    
    transform_image = image_transform()

    with torch.no_grad():
        for i, file in enumerate(tqdm(files)):
            try:
                img = Image.open(file)
                transformed_img = transform_image(img)[:3].unsqueeze(0).to(device)
                embeddings = model(transformed_img, return_register_tokens=True)
                patch_tokens, class_tokens, register_tokens = embeddings[0]
                all_embeddings[file] = {
                    "patch_tokens": patch_tokens.cpu().numpy().tolist(),
                    "class_tokens": class_tokens.cpu().numpy().tolist(),
                    "register_tokens": register_tokens.cpu().numpy().tolist()
                }
            except Exception as e:
                print(f"Error processing {file}: {e}")
                excluded_files.append(file)
    
    return all_embeddings, excluded_files

def save_embeddings(embeddings, save_dir):
    """
    Save embeddings and metadata to the specified directory.
    
    Args:
        embeddings (dict): Dictionary where keys are filenames and values are embeddings.
        save_dir (str): Directory to save the embeddings and metadata.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save each embedding as a separate .npy file
    for filename, embedding in embeddings.items():
        np.save(os.path.join(save_dir, f"{os.path.splitext(os.path.basename(filename))[0]}.npy"), embedding)
    
    # Save metadata as JSON
    metadata = {filename: os.path.splitext(os.path.basename(filename))[0] + '.npy' for filename in embeddings.keys()}
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
