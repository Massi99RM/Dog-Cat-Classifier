"""
Creates a PyTorch Dataset that loads images from folders, resizes them, and applies transforms (normalization, augmentation).
Handles loading images from disk and applying the necessary transformations to prepare them for training and validation.
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Image dimensions expected by ResNet18
IMAGE_SIZE = 224

# Normalization values used by ImageNet-pretrained models
# These are the mean and standard deviation of the ImageNet dataset
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    """
    Returns the transformation pipeline for training images.
    Training transforms include data augmentation techniques that help
    the model generalize better by showing it variations of each image.
    """
    return transforms.Compose([
        # Resize to slightly larger than target, then crop randomly
        # Adds positional variation to the training data
        transforms.Resize(256),
        transforms.RandomCrop(IMAGE_SIZE),
        
        # Randomly flip images horizontally (a cat facing left is still a cat)
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Small color variations to make the model ready to lighting changes
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        
        # Convert PIL Image to PyTorch tensor (values become 0-1 range)
        transforms.ToTensor(),
        
        # Normalize using ImageNet statistics (necessary for pretrained models)
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_validation_transforms():
    """
    Returns the transformation pipeline for validation images.
    Validation transforms are simpler - no augmentation, just resize and normalize.
    It gives consistent results when evaluating the model.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Creates DataLoader objects for training and validation datasets.
    Args:
        data_dir: Path to the data folder containing 'train' and 'validate' subfolders
        batch_size: Number of images to process at once
        num_workers: Number of CPU threads for loading data
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        class_names: List of class names [cats, dogs]
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validate')
    
    # ImageFolder automatically assigns labels based on subfolder names
    # Images in 'cats' folder get label 0, images in 'dogs' folder get label 1
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transforms()
    )
    
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=get_validation_transforms()
    )
    
    # DataLoader handles batching, shuffling, and parallel loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Randomize order each epoch for better training
        num_workers=num_workers,
        pin_memory=True  # Speeds up CPU to GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get class names from folder structure
    class_names = train_dataset.classes
    
    return train_loader, val_loader, class_names


if __name__ == '__main__':
    # Quick test to verify the dataset loads correctly
    train_loader, val_loader, class_names = create_dataloaders('data', batch_size=4)
    
    print(f"Classes found: {class_names}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Load one batch to verify everything works
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")