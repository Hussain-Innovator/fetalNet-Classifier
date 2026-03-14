import streamlit as st
import os
from utils.prediction import predict_image  # your existing prediction function
from PIL import Image

st.title("FetalNet Public Test Images")

# Path to test images folder
test_folder = "test_images"
image_files = [f for f in os.listdir(test_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

if not image_files:
    st.warning("No test images found in the folder!")
else:
    st.write(f"Found {len(image_files)} images for testing.")

    for image_name in image_files:
        image_path = os.path.join(test_folder, image_name)
        image = Image.open(image_path)
        st.image(image, caption=image_name, use_column_width=True)

        # Make prediction using your existing function
        result = predict_image(image_path)
        st.success(f"Prediction: {result}")