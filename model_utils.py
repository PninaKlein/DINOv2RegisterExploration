import torch
import torchvision.transforms as T
import os

def initialize_model(device):
    """
    Load and initialize the DINOv2 model.
    
    Args:
        device (torch.device): Device to load the model on (CPU or GPU).
    
    Returns:
        model: Loaded DINOv2 model.
    """
    model = torch.hub.load(os.getcwd(), 'dinov2_vits14_reg', source='local')
    model.to(device)
    return model

def image_transform():
    """
    Define image transformations to prepare images for the DINOv2 model.
    
    Returns:
        transform (torchvision.transforms.Compose): Transformation pipeline for images.
    """
    return T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])
