"""
Defines the CNN architecture using transfer learning.
A pretrained ResNet18 model it's used and only the final classification layer is modified.
"""

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=2, pretrained=True):
    """
    Creates a ResNet18 model configured for binary classification.
    Transfer learning approach:
    1. Load ResNet18 with weights trained on ImageNet
    2. Freeze the early layers (they already know how to extract features)
    3. Replace the final layer to output the 2 classes (cats, dogs)
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
    Returns:
        model: Configured PyTorch model ready for training
    """
    if pretrained:
        # Load ResNet18 with pretrained ImageNet weights
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        # Load ResNet18 without pretrained weights (random initialization)
        model = models.resnet18(weights=None)
    
    # Freeze all layers initially
    # Frozen layers won't be updated during training, saving computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    # Original ResNet18 outputs 1000 classes, only 2 needed for cats and dogs
    # The 'fc' layer takes 512 features from the previous layer
    num_features = model.fc.in_features
    
    # Create a new classification head
    # This is the only part trained from scratch
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # Helps prevent overfitting
        nn.Linear(num_features, num_classes)
    )
    
    return model


def unfreeze_model(model, unfreeze_from='layer4'):
    """
    Unfreezes later layers of the model for fine-tuning.
    After initial training with frozen layers, it's optional to unfreeze
    some layers to fine-tune them on a specific dataset. This can improve
    accuracy but requires careful learning rate adjustment.
    Args:
        model: The ResNet18 model
        unfreeze_from: Which layer to start unfreezing from 'layer1', 'layer2', 'layer3', 'layer4', 'all'
    """
    if unfreeze_from == 'all':
        for param in model.parameters():
            param.requires_grad = True
        return
    
    # Map layer names to their position in the unfreezing order
    layer_order = ['layer1', 'layer2', 'layer3', 'layer4']
    
    if unfreeze_from not in layer_order:
        raise ValueError(f"unfreeze_from must be one of {layer_order} or 'all'")
    
    start_idx = layer_order.index(unfreeze_from)
    layers_to_unfreeze = layer_order[start_idx:]
    
    for name, param in model.named_parameters():
        for layer_name in layers_to_unfreeze:
            if layer_name in name:
                param.requires_grad = True


def count_parameters(model):
    """
    Counts the total and trainable parameters in the model.
    Needed for understanding the model size and verifying that the
    layer freezing idea is working correctly.
    Returns:
        total: Total number of parameters
        trainable: Number of parameters that will be updated during training
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    # Quick test to verify the model loads correctly
    model = create_model(num_classes=2, pretrained=True)
    
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {total - trainable:,}")
    
    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")