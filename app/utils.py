"""
Utility functions and preprocessing helpers
"""

from torchvision import transforms
from PIL import Image
import numpy as np


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_transforms(train=True):
    """
    Get data transforms for training or inference
    
    Args:
        train: Whether to return training transforms (with augmentation)
        
    Returns:
        torchvision.transforms.Compose object
    """
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def preprocess_image(image):
    """
    Preprocess image for model input
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        Preprocessed tensor
    """
    pass


def postprocess_predictions(outputs):
    """
    Convert model outputs to human-readable predictions
    
    Args:
        outputs: Model output tensor
        
    Returns:
        Dictionary with class names and probabilities
    """
    pass
