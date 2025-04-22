import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.graph_objects as go

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
model = AutoModelForSequenceClassification.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")

# Streamlit UI
st.title("Spam Detector")

input_text = st.text_area("Enter text to analyze:", height=150)

if st.button("Detect"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Tokenize the input text
        inputs = tokenizer([input_text], return_tensors="pt", padding=True, truncation=True)

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=1)

        # Get predicted labels
        predictions = torch.argmax(probabilities, dim=1)

        # Map labels to class names
        label_map = {0: "Not Spam", 1: "Spam"}

        # Display the prediction and dynamic graph
        pred_label = label_map[predictions.item()]
        confidence = probabilities[0][predictions.item()].item()

        # Display Result with dynamic text
        st.subheader("Result")
        st.write(f"**Label:** {pred_label} (Confidence: {confidence:.2f})")

        # Dynamic Warning for Spam
        if pred_label == "Spam":
            st.error("Warning: This message is detected as Spam! Avoid clicking on suspicious links or providing sensitive information.")
        else:
            st.success("This message is not detected as spam. It's safe!")

        # Plotly Interactive Bar Chart for Class Probabilities
        fig = go.Figure()

        # Add bars for the classes (Spam and Not Spam)
        fig.add_trace(go.Bar(
            x=list(label_map.values()),
            y=probabilities[0].detach().numpy(),
            text=probabilities[0].detach().numpy(),
            textposition='auto',
            marker=dict(color=['red', 'green']),
            name="Probability"
        ))

        fig.update_layout(
            title="Spam Detection Probability",
            xaxis_title="Class",
            yaxis_title="Probability",
            template="plotly_dark",
            showlegend=False,
            hovermode="closest"
        )

        # Display the interactive graph
        st.plotly_chart(fig)
