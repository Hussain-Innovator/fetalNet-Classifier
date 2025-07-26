import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page configuration
st.set_page_config(page_title="Model Performance", page_icon="", layout="wide")

st.title("Model Performance: Baseline vs. Fine-Tuned ")
st.markdown("This page compares the performance of the initial model against the final, fine-tuned model.")

# --- Data for summary tables ---
baseline_data = {
    "Metric": ["Best Validation Accuracy", "Final Test Accuracy", "F1-score (Macro Avg)", "F1-score (Weighted)"],
    "Score": ["92.69% (Epoch 5)", "91.77%", "0.90", "0.92"]
}
finetuned_data = {
    "Metric": ["Best Validation Accuracy", "Final Test Accuracy", "F1-score (Macro Avg)", "F1-score (Weighted)"],
    "Score": ["93.34% (Epoch 10)", "93.01%", "0.91", "0.93"]
}
df_baseline = pd.DataFrame(baseline_data)
df_finetuned = pd.DataFrame(finetuned_data)

# --- Display Side-by-Side Summary Tables ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Baseline Model")
    st.table(df_baseline)
with col2:
    st.subheader("Fine-Tuned Model")
    st.table(df_finetuned)

st.markdown("---")

st.header("Full Training & Fine-Tuning History (20 Epochs)")

history_data = {
    'train_acc': [83.95, 90.13, 92.44, 94.24, 95.09, 96.14, 97.05, 97.64, 97.72, 97.89, 96.88, 97.94, 98.26, 98.48, 98.26, 98.93, 98.86, 99.10, 99.15, 99.18],
    'val_acc': [90.22, 92.05, 91.99, 91.72, 92.69, 91.24, 92.37, 92.53, 91.94, 92.21, 92.80, 92.91, 93.18, 92.80, 93.18, 93.01, 92.91, 93.07, 93.18, 93.34],
    'train_loss': [0.7467, 0.3326, 0.2427, 0.1886, 0.1545, 0.1205, 0.0967, 0.0821, 0.0689, 0.0680, 0.1060, 0.0801, 0.0691, 0.0644, 0.0619, 0.0538, 0.0507, 0.0424, 0.0403, 0.0424],
    'val_loss': [0.3432, 0.2513, 0.2524, 0.2541, 0.2328, 0.2823, 0.2410, 0.2548, 0.2900, 0.2715, 0.2118, 0.2140, 0.2177, 0.2215, 0.2214, 0.2229, 0.2230, 0.2330, 0.2249, 0.2290]
}
epochs_range = range(1, 21)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: Loss History
ax1.plot(epochs_range, history_data['train_loss'], 'o-', label='Training Loss')
ax1.plot(epochs_range, history_data['val_loss'], 'o-', label='Validation Loss')
ax1.set_title('Combined Loss History')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.axvline(x=10, color='r', linestyle='--', linewidth=2, label='Fine-Tuning Started')
ax1.legend()
ax1.grid(True)

# Plot 2: Accuracy History
ax2.plot(epochs_range, history_data['train_acc'], 'o-', label='Training Accuracy')
ax2.plot(epochs_range, history_data['val_acc'], 'o-', label='Validation Accuracy')
ax2.set_title('Combined Accuracy History')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.axvline(x=10, color='r', linestyle='--', linewidth=2, label='Fine-Tuning Started')
ax2.legend()
ax2.grid(True)

# Display the plot in Streamlit
st.pyplot(fig)


st.markdown("---")

# --- This is your existing F1-Score graph code ---
st.header("Class-wise F1-Score Comparison")

f1_scores_initial = { 'Fetal abdomen': 0.77, 'Fetal brain': 0.97, 'Fetal femur': 0.87, 'Fetal thorax': 0.86, 'Maternal cervix': 1.00, 'Other': 0.90 }
f1_scores_finetuned = { 'Fetal abdomen': 0.78, 'Fetal brain': 0.98, 'Fetal femur': 0.88, 'Fetal thorax': 0.90, 'Maternal cervix': 1.00, 'Other': 0.92 }

labels = list(f1_scores_initial.keys())
initial_scores = list(f1_scores_initial.values())
finetuned_scores = list(f1_scores_finetuned.values())

x = np.arange(len(labels))
width = 0.35

fig2, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, initial_scores, width, label='Initial Run', color='skyblue')
rects2 = ax.bar(x + width/2, finetuned_scores, width, label='Fine-Tuned Run', color='sandybrown')

ax.set_ylabel('F1-Score')
ax.set_title('Class-wise F1-Score Comparison: Initial vs. Fine-Tuned')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

fig2.tight_layout()
st.pyplot(fig2)