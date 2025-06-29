# pages/3_Project_Overview.py
import streamlit as st
from PIL import Image # Import the Image library

st.set_page_config(page_title="Project Overview", page_icon="", layout="wide")

st.title("Project Overview")

st.markdown("""
This project presents an end-to-end deep learning solution for the classification of fetal ultrasound images. The goal is to accurately identify the anatomical plane shown in an ultrasound scan, which is a critical step in prenatal diagnosis.

### Key Objectives:
- To build a robust image classification model using modern deep learning techniques.
- To use transfer learning from a pre-trained model (`MobileNetV2`) to achieve high accuracy.
- To enhance the model with an attention mechanism (SE Block).
- To systematically evaluate the model's performance and understand the contribution of its components through ablation studies.
- To deploy the final model in an easy-to-use, interactive web application using Streamlit.

### Final Model Performance:
- **Architecture:** `FetalNet` (MobileNetV2 + SE Block + Custom Layers)
- **Best Test Accuracy:** **93.01%**
- **Best Macro F1 Score:** **0.91**
""")

# --- NEW: Code to display a local image ---
try:
    # Open the image file from your assets folder
    image = Image.open("app_assets/overview_image.png")
    # Display the image
    st.image(image, caption="Example of Fetal Ultrasound Scans Used in This Project")
except FileNotFoundError:
    st.warning("Could not find the example image. Make sure it's in the 'app_assets' folder.")

# --- NEW: Section to display the model architecture diagram ---
st.header("Model Architecture")
st.markdown("Our model classifies fetal ultrasound images into six anatomical planes using a MobileNetV2-based deep learning model enhanced with a squeeze-and-excitation (SE) block. To ensure transparency, it integrates Explainable AI techniques—Grad-CAM, Saliency Map, Integrated Gradients, and Guided Backpropagation. The model is deployed as an interactive web app using Streamlit for real-time predictions and visual explanations")

try:
    # Open the diagram image file from your assets folder
    image = Image.open("app_assets/model_architecture.PNG")
    # Display the image
    st.image(image, caption="Block Diagram of the FetalNet Model Architecture")
except FileNotFoundError:
    st.warning("Could not find the model architecture diagram. Make sure 'model_architecture.png' is in the 'app_assets' folder.")


st.markdown("---")
