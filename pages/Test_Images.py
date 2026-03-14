import streamlit as st
import os
from utils.prediction import predict_image  # assuming you have a function for predictions

st.title("Public Test Images")

# Path to test images
test_folder = "test_images"
image_files = [f for f in os.listdir(test_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

# Show all test images with predictions
for image_name in image_files:
    image_path = os.path.join(test_folder, image_name)
    st.image(image_path, caption=image_name, use_column_width=True)
    
    # Get prediction
    result = predict_image(image_path)  # adapt this function if needed
    st.write("Prediction:", result)