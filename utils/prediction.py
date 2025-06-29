# utils/prediction.py
import torch
import streamlit as st
from model.fetalnet import FetalNet

@st.cache_resource
def load_model(model_path):
    """Loads the trained FetalNet model from the specified path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FetalNet(num_classes_model=6)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure the model is in the correct directory.")
        return None

@torch.no_grad()
def get_prediction(model, image_tensor):
    """
    Takes a model and a tensor.
    
    Returns:
    - The top predicted class index.
    - The confidence score for the top class.
    - A NumPy array of probabilities for ALL classes.
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Get the top prediction and its confidence
    confidence, predicted_idx = torch.max(probabilities, 1)

    # --- CHANGED HERE: Return all probabilities as well ---
    return predicted_idx.item(), confidence.item(), probabilities.squeeze().cpu().numpy()















# # utils/prediction.py
# import torch
# import streamlit as st
# from model.fetalnet import FetalNet # Import your main model class

# @st.cache_resource
# def load_model(model_path):
#     """Loads the trained FetalNet model from the specified path."""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = FetalNet(num_classes_model=6)
#     try:
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         model.to(device)
#         model.eval()
#         return model
#     except FileNotFoundError:
#         st.error(f"Model file not found at {model_path}. Please ensure the model is in the correct directory.")
#         return None

# @torch.no_grad()
# def get_prediction(model, image_tensor):
#     """Takes a model and a tensor, returns the predicted class, index, and confidence."""
#     device = next(model.parameters()).device
#     image_tensor = image_tensor.to(device)

#     outputs = model(image_tensor)
#     probabilities = torch.nn.functional.softmax(outputs, dim=1)
#     confidence, predicted_idx = torch.max(probabilities, 1)

#     return predicted_idx.item(), confidence.item()