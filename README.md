# Fetal Ultrasound Plane Classifier with Explainable AI

This repository contains the complete code and trained model for a Final Year Project focused on classifying fetal ultrasound images. The project culminates in an interactive web application built with Gradio that not only predicts the anatomical plane but also provides multiple Explainable AI (XAI) visualizations to interpret the model's decisions.

## Project Overview

This project addresses the critical need for automated classification of fetal ultrasound planes, a key task in prenatal diagnostics. Manual interpretation of these images is time-consuming and dependent on operator experience. The primary objective was to develop a deep learning model to accurately classify ultrasound images into six categories:

- Fetal abdomen
- Fetal brain
- Fetal femur
- Fetal thorax
- Maternal cervix
- Other

## Key Features

- **Live Classification:** Upload a fetal ultrasound image and get an instant prediction of its anatomical plane.
- **Detailed Probabilities:** View the model's confidence score for all six possible classes.
- **Explainable AI (XAI) Dashboard:** An interactive, tabbed interface that visualizes the model's reasoning using four different techniques:
  - **Grad-CAM:** Highlights the important regions the model focused on.
  - **Guided Backpropagation:** Shows the specific pixels and edges that influenced the decision.
  - **Integrated Gradients:** A stable method for attributing importance to pixels.
  - **Occlusion Sensitivity:** A test that shows which parts of the image are most critical to the prediction.

## Model and Performance

The core of this project is `FetalNet`, a custom deep learning architecture implemented in PyTorch.

- **Architecture:** It uses a frozen MobileNetV2 backbone for efficient feature extraction, enhanced with a custom head that includes a Squeeze-and-Excitation (SE) Block for attention.
- **Final Performance:** The model was optimized through a two-phase training and fine-tuning process. The final, best-performing model achieved the following results on the unseen test set:
  - **Test Accuracy:** 93.01%
  - **Weighted F1-Score:** 0.93

## How to Run Locally

To run this application on your local machine, please follow these steps. It is recommended to use Conda for managing the environment.

### 1. Clone the repository

```bash
git clone https://github.com/Hussain-Innovator/fetalNet-Classifier.git
cd fetalNet-Classifier

3. Install dependencies

# Pull the large model file
git lfs pull

# Install Python packages
pip install -r requirements.txt
4. Run the Gradio app

Project Structure

├── app.py                         
├── requirements.txt               
├── README.md                      
├── .gitignore
├── .gitattributes
├── model/
│   ├── fetalnet.py                # FetalNet model architecture
│   └── trained_models/
│       └── best_model.pth         # Saved model weights
├── utils/
│   ├── preprocess.py              # Image transformation logic
│   ├── prediction.py              # Model inference logic
│   └── xai.py                     # XAI visualization functions
├── pages/
│   ├── 1_Model_Performance.py     # Tab: model performance metrics
│   ├── 2_Ablation_Study.py        # Tab: ablation study analysis
│   └── 3_Project_Overview.py      # Tab: general project overview
└── app_assets/
    ├── model_architecture.PNG     # Visual of the model
    └── overview_image.png         # Summary or dashboard image


## Author

**Hussain**  
BS Software Engineering  
Iqra University, 2021–2025  
[GitHub](https://github.com/Hussain-Innovator)  
[Email](mailto:hussainsamdaniS686@gmail.com)
