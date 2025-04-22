import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import plotly.express as px
from safetensors.torch import load_file  # Optional if you're using .safetensors

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./ai_detector")
    model_path = "./ai_detector"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model from .bin: {e}")
        model_weights = load_file(f"{model_path}/model.safetensors")
        model = AutoModelForSequenceClassification.from_pretrained(model_path, state_dict=model_weights)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("🧠 AI Text Detector")
st.markdown("Analyze a piece of text to check if it was written by a human or AI.")

input_text = st.text_area("✍️ Enter text to analyze:", height=150)

if st.button("Detect"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        pred_label = torch.argmax(predictions).item()
        confidence_scores = predictions[0].tolist()

        labels = ["Human-written", "AI-generated"]
        confidence = confidence_scores[pred_label]

        # 🎯 Display result text
        result_text = f"### 🏷️ **Result:** {labels[pred_label]}  \n"
        result_text += f"**Confidence:** `{confidence:.2%}`"
        st.markdown(result_text)

        # 📊 Plot a confidence bar chart
        df = pd.DataFrame({
            "Label": labels,
            "Confidence": confidence_scores
        })
        fig = px.bar(df, x="Label", y="Confidence", color="Label", 
                     range_y=[0,1], text=df["Confidence"].map(lambda x: f"{x:.2%}"),
                     color_discrete_sequence=["#2ecc71", "#e74c3c"])
        fig.update_traces(textposition='outside')
        fig.update_layout(title="Prediction Confidence", yaxis_title="Confidence", xaxis_title="Label")

        st.plotly_chart(fig)

        # 💬 Verdict Message
        if pred_label == 0:
            st.success("✅ The model thinks this text is **likely written by a human**.")
        else:
            st.error("🤖 This text appears to be **AI-generated** based on the model's prediction.")
