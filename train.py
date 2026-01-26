"""
Main training script: loads data, runs the training loop, saves the best model and the final model.
It handles the complete training pipeline:
1. Load and prepare the data
2. Initialize the model
3. Run the training loop
4. Validate after each epoch
5. Save the best performing model
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import create_dataloaders
from model import create_model, count_parameters


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Trains the model for one complete round through the training data.
    Args:
        model: The neural network
        train_loader: DataLoader containing training data
        criterion: Loss function using CrossEntropyLoss
        optimizer: Optimization algorithm using Adam
        device: 'cuda' for GPU or 'cpu'
    Returns:
        epoch_loss: Average loss across all batches
        epoch_acc: Accuracy as a percentage
    """
    model.train()  # Set model to training mode (enables dropout)
    
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to GPU if available
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the gradients from previous batch
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / total_samples
    epoch_acc = 100 * correct_predictions / total_samples
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Training stopped, no gradient computation needed.
    Args:
        model: The neural network
        val_loader: DataLoader containing validation data
        criterion: Loss function
        device: 'cuda' for GPU or 'cpu'
    Returns:
        val_loss: Average loss on validation set
        val_acc: Accuracy as a percentage
    """
    model.eval()  # Set model to evaluation mode (disables dropout, used only with training images)
    
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    val_loss = running_loss / total_samples
    val_acc = 100 * correct_predictions / total_samples
    
    return val_loss, val_acc


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_history.png'):
    """
    Creates and saves plots showing training progress over epochs, helps finding eventual training issues.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")


def train(data_dir='data', num_epochs=10, batch_size=32, learning_rate=0.001, save_dir='checkpoints'):
    """
    Main training function.
    Args:
        data_dir: Path to data folder with train/validate subfolders
        num_epochs: Number of complete passes through the training data
        batch_size: Number of images per batch
        learning_rate: Step size for optimization
        save_dir: Directory to save model checkpoints
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Detect if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    # num_workers=0 on Windows to avoid multiprocessing issues
    num_workers = 0 if os.name == 'nt' else 4
    train_loader, val_loader, class_names = create_dataloaders(
        data_dir, 
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    model = create_model(num_classes=len(class_names), pretrained=True)
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    # Parameters that require gradients are the only ones optimized (the unfrozen ones)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    # # Learning rate scheduler, it reduces learning rate when validation loss stops improving/plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2
    )
    
    # Track training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)
    
    total_start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Epoch time: {epoch_time:.1f}s")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': class_names
            }, best_model_path)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
    
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Total training time: {total_time / 60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save the final model
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'class_names': class_names
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    return model


if __name__ == '__main__':
    # The parameters can be modified here
    train(
        data_dir='data',
        num_epochs=10,
        batch_size=32,
        learning_rate=0.001,
        save_dir='checkpoints'
    )