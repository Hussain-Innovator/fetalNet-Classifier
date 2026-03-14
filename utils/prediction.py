# utils/prediction.py
import torch
from model.fetalnet import FetalNet
import streamlit as st
from torchvision import transforms
from PIL import Image
import numpy as np

@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FetalNet(num_classes_model=6)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}.")
        return None

@torch.no_grad()
def get_prediction(model, image_tensor):
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)
    return predicted_idx.item(), confidence.item(), probabilities.squeeze().cpu().numpy()

def predict_image(model, image_path):
    """Wrapper to load an image and predict using the model."""
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return get_prediction(model, image_tensor)