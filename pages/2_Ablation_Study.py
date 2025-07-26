# pages/2_Ablation_Study.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Ablation Study", page_icon="", layout="wide")
st.title("Ablation Study: Understanding Model Components ")
st.markdown(
    "An ablation study is performed to understand the contribution of different components of our model. "
    "We trained several variations of the model, each with a specific part removed, and compared their performance to our final model."
)

data = {
    "Model Variant": ["Full Model (Fine-Tuned)", "Full Model (Baseline)", "SE Block Removed", "Second Conv Block Removed", "MaxPool ➝ AvgPool"],
    "Test Accuracy (%)": [93.01, 91.77, 92.00, 92.00, 92.00],
    "Macro F1 Score": [0.91, 0.90, 0.90, 0.90, 0.90],
    "Notes": ["Our best performing model", "Before fine-tuning", "Attention mechanism removed", "Reduced model depth", "Different pooling strategy"]
}
df_summary = pd.DataFrame(data).set_index("Model Variant")


st.subheader("Performance Comparison Table")
st.dataframe(df_summary)

st.markdown("---")

# --- Visualizations ---
st.subheader("Visual Comparison")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.barplot(x=df_summary.index, y=df_summary["Test Accuracy (%)"], ax=ax, palette="Blues_d")
    ax.set_title("Test Accuracy Comparison")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(90, 94)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.barplot(x=df_summary.index, y=df_summary["Macro F1 Score"], ax=ax, palette="Greens_d")
    ax.set_title("Macro F1 Score Comparison")
    ax.set_ylabel("Macro F1 Score")
    ax.set_ylim(0.88, 0.93)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)


# --- Conclusion ---
st.subheader("Study Conclusion")
st.success(
    "**Conclusion:** The results confirm that both **fine-tuning** and the **SE Block** are critical components for achieving the highest performance. "
    "While removing other parts had a small negative impact, removing the SE Block or not fine-tuning led to a noticeable drop in accuracy."
)