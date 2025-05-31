import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Model Metrics Summary ---
# --- Model Metrics Summary ---
model_metrics = {
    'DFDC Deepfake Detection': {
        'Accuracy': 92.02,
        'Precision': 65.0,
        'Recall': 50.0,
        'F1-Score': 56.0,
        'AUC-ROC': 97.0
    },
    'AI Detector': {
        'Accuracy': 89.6,
        'Precision': 88.2,
        'Recall': 87.9,
        'F1-Score': 88.0,
        'AUC-ROC': 91.5
    },
    'Spam Detector': {
        'Accuracy': 95.3,
        'Precision': 94.8,
        'Recall': 95.0,
        'F1-Score': 94.9,
        'AUC-ROC': 96.2
    },
    'Multimodal Misinformation Detection': {
        'Accuracy': 91.94,
        'Precision': 93.76,
        'Recall': 93.29,
        'F1-Score': 93.29,
        'AUC-ROC': None
    }
}


# --- Multimodal Classifier Report ---
report_dict = {
    'Class': ['True', 'Satire', 'False Conn', 'Imposter', 'Manipulated', 'Misleading'],
    'Precision': [0.89, 0.83, 0.78, 0.80, 0.88, 0.80],
    'Recall': [0.91, 0.64, 0.77, 0.41, 0.93, 0.80],
    'F1-Score': [0.90, 0.72, 0.77, 0.54, 0.90, 0.80],
    'Support': [1251, 124, 388, 39, 622, 75]
}
report_df = pd.DataFrame(report_dict)

# --- App Title ---
st.title("📊 Model Performance Dashboard")

# --- Model Selector ---
selected_model = st.selectbox("Choose a model to view its metrics:", list(model_metrics.keys()))
metrics = model_metrics[selected_model]
df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']).dropna()

# --- Display Model Bar Chart ---
if not df.empty:
    st.subheader(f"{selected_model} Metrics")
    st.bar_chart(df)

    # --- Show DFDC Loss Plot ---
    if selected_model == "DFDC Deepfake Detection":
        st.markdown("### 📉 Training Loss Curve")
        st.image("./loss_plot_dfdc.png", caption="Weighted loss over epochs")

else:
    st.warning(f"No available metrics to display for **{selected_model}**.")

# --- Multimodal Specific Metrics ---
if selected_model == "Multimodal Misinformation Detection":
    st.subheader("📊 Multimodal Misinformation Classifier Metrics")
    metric_option = st.selectbox("Choose metric to visualize:", ["Classification Table", "Confusion Matrix", "Prediction Histogram"])

    if metric_option == "Classification Table":
        st.markdown("### 🔍 Classification Report")
        st.dataframe(report_df.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}"}))

    elif metric_option == "Confusion Matrix":
        cm_data = [
            [1140, 7, 39, 2, 62, 1],
            [8, 79, 16, 1, 16, 4],
            [17, 5, 298, 2, 61, 5],
            [2, 0, 2, 16, 19, 0],
            [18, 4, 13, 2, 580, 5],
            [1, 3, 2, 0, 9, 60]
        ]
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='coolwarm',
                    xticklabels=report_df['Class'], yticklabels=report_df['Class'])
        plt.title("Confusion Matrix")
        st.pyplot(plt)

    elif metric_option == "Prediction Histogram":
        plt.figure(figsize=(8, 4))
        sns.barplot(x=report_df['Class'], y=report_df['Support'])
        plt.title("Prediction Count by Class")
        plt.xticks(rotation=45)
        st.pyplot(plt)
st.markdown("""
    <style>
    div[data-baseweb="select"] > div {
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)