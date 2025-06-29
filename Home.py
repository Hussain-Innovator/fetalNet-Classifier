import streamlit as st
import pandas as pd
from PIL import Image
from utils.preprocess import process_image_for_prediction
from utils.prediction import load_model, get_prediction
from utils.xai import explain_prediction

# --- Page Configuration ---
st.set_page_config(page_title="Fetal Ultrasound AI", page_icon="", layout="wide")

# --- Header ---
st.title("FetalNet Ultrasound Classifier 🔬")
st.markdown("Upload an image to classify the fetal plane and explore different AI explanations.")

# --- Model Loading ---
MODEL_PATH = "model/trained_models/best_model.pth"
model = load_model(MODEL_PATH)
CLASS_NAMES = ['Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax', 'Maternal cervix', 'Other']

# --- Sidebar ---
st.sidebar.title("About the Model")
st.sidebar.info("The model uses MobileNetV2’s pretrained feature extractor combined with additional convolutional layers and an SE (Squeeze-and-Excitation) block. It has been fine-tuned on fetal ultrasound images to accurately classify them into six anatomical categories. The SE block helps the model focus on the most important features in each image, improving overall performance without adding much complexity.")
st.sidebar.success(f"Final Validation Accuracy: **93.34%**")
st.sidebar.success(f"Final Test Accuracy: **93.01%**")


# --- Image Uploader and Prediction ---
if model:
    uploaded_file = st.file_uploader("Upload an ultrasound image to analyze:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([0.5, 0.5])
        original_image = Image.open(uploaded_file).convert("RGB")

        # --- LEFT COLUMN: For the uploaded image ---
        with col1:
            st.subheader("Input Image")
            st.image(original_image, caption="Image to be analyzed", use_container_width=True)

        # --- RIGHT COLUMN: For the button and all results ---
        with col2:
            st.subheader("Analysis Controls & Results")
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing and generating explanations..."):
                    image_tensor = process_image_for_prediction(uploaded_file)
                    
                    pred_idx, confidence, all_probs = get_prediction(model, image_tensor)
                    predicted_class = CLASS_NAMES[pred_idx]

                    st.success(f"**Predicted Plane:** {predicted_class} ({confidence*100:.2f}% confidence)")
                    st.markdown("---")

                    st.write("##### All Class Probabilities")
                    prob_df = pd.DataFrame({
                        "Class": CLASS_NAMES,
                        "Probability": all_probs
                    })
                    st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)
                    st.markdown("---")
                    
                    st.write("##### Model Explanation (XAI) Comparison")
                    tab1, tab2, tab3, tab4 = st.tabs(["Grad-CAM", "Saliency Map", "Integrated Gradients", "Guided Backpropagation"])

                    with tab1:
                        st.info("**Grad-CAM:** Highlights broad, class-discriminative regions.")
                        viz = explain_prediction(model, image_tensor.clone(), original_image, pred_idx, method='Grad-CAM')
                        # CHANGED HERE
                        st.image(viz, caption="Grad-CAM Explanation", use_container_width=True)

                    with tab2:
                        st.info("**Saliency Map:** Shows the pixels that most influence the prediction.")
                        viz = explain_prediction(model, image_tensor.clone(), original_image, pred_idx, method='Saliency')
                        # CHANGED HERE
                        st.image(viz, caption="Saliency Map Explanation", use_container_width=True)

                    with tab3:
                        st.info("**Integrated Gradients:** A more robust method that reduces noise.")
                        viz = explain_prediction(model, image_tensor.clone(), original_image, pred_idx, method='Integrated Gradients')
                        # CHANGED HERE
                        st.image(viz, caption="Integrated Gradients Explanation", use_container_width=True)

                    with tab4:
                        st.info("**Guided Backpropagation:** Produces fine-grained, high-resolution visualizations.")
                        viz = explain_prediction(model, image_tensor.clone(), original_image, pred_idx, method='Guided Backpropagation')
                        # CHANGED HERE
                        st.image(viz, caption="Guided Backpropagation Explanation", use_container_width=True)
else:
    st.warning("The model could not be loaded. Please check the file path in the code.")
