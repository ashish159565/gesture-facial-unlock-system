import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
])

# Function to create directory structure for ImageFolder
def setup_data_directory(base_dir="data/gestures"):
    # Create directories if they don't exist
    os.makedirs(os.path.join(base_dir, "peacesign"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "other"), exist_ok=True)
    print(f"Created directories at {base_dir}")
    print(f"Place peace sign images in: {os.path.join(base_dir, 'peacesign')}")
    print(f"Place other gesture images in: {os.path.join(base_dir, 'other')}")

# Function to display some sample images from the dataset
def display_samples(dataloader, class_names):
    images, labels = next(iter(dataloader))
    plt.figure(figsize=(10, 8))
    for i in range(min(16, len(images))):
        plt.subplot(4, 4, i+1)
        # Convert tensor to image
        img = images[i].numpy().transpose((1, 2, 0))
        # Denormalize
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(class_names[labels[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Set up directory structure
setup_data_directory()

# Load dataset (will work once you've added images to the folders)
try:
    dataset = ImageFolder(root="data/gestures", transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"Dataset loaded successfully with {len(dataset)} images")
    print(f"Class names: {dataset.classes}")
    
    # Display sample images
    display_samples(dataloader, dataset.classes)
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Make sure you've added images to the data/gestures/peace_sign and data/gestures/other directories")