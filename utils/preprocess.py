# utils/preprocess.py
import torch
import torchvision.transforms as transforms
from PIL import Image

def process_image_for_prediction(image_bytes):
    """Takes image bytes, applies transformations, and returns a tensor."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3) # Using the normalization from your notebook
    ])
    image = Image.open(image_bytes).convert("RGB")
    return transform(image).unsqueeze(0)