"""
Loads a trained model and classifies new images.
Used to test the model on images it has never seen before.
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms

from model import create_model


# Same normalization values used during training
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_prediction_transforms():
    """
    Returns the transformation pipeline for prediction.
    Must match the validation transforms used during training.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_model(checkpoint_path, device):
    """
    Loads a trained model from a checkpoint file.
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: 'cuda' or 'cpu'
    Returns:
        model: The loaded model ready for inference
        class_names: List of class names ['cats', 'dogs']
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get class names from checkpoint
    class_names = checkpoint.get('class_names', ['cats', 'dogs'])
    
    # Create model architecture
    model = create_model(num_classes=len(class_names), pretrained=False)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    
    return model, class_names


def predict_image(image_path, model, class_names, device):
    """
    Classifies a single image.
    Args:
        image_path: Path to the image file
        model: Loaded PyTorch model
        class_names: List of class names
        device: 'cuda' or 'cpu'
    Returns:
        predicted_class: The predicted class name ('cats' or 'dogs')
        confidence: Confidence score as a percentage
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    transform = get_prediction_transforms()
    input_tensor = transform(image)
    
    # Add batch dimension (model expects batch of images)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_batch)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the predicted class and confidence
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted_idx.item()]
        confidence_percent = confidence.item() * 100
    
    return predicted_class, confidence_percent


def main():
    """
    Main function that parses command line arguments and runs prediction.
    """
    parser = argparse.ArgumentParser(
        description='Classify images as cat or dog using a trained model.'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to one image to classify'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to the trained model checkpoint (default: checkpoints/best_model.pth)'
    )
    
    args = parser.parse_args()
    
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    model, class_names = load_model(args.model, device)
    
    # Run prediction
    predicted_class, confidence = predict_image(args.image, model, class_names, device)
    animal = predicted_class.rstrip('s')
    print(f"\nPrediction: {animal.upper()}")
    print(f"Confidence: {confidence:.1f}%")


if __name__ == '__main__':
    main()