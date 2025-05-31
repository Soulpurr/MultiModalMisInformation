# Import necessary libraries
import streamlit as st
import subprocess
import numpy as np
import os
import uuid
import tempfile
import torch
import re
import matplotlib.pyplot as plt
import sys
import plotly.express as px
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safetensors.torch import load_file  # for .safetensors
import plotly.graph_objects as go
from image_text.models.BertNetClassifier import BERTResNetClassifier
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
import torch.nn.functional as F
import seaborn as sns
# ----- Local Imports -----
sys.path.append(os.path.join(os.path.dirname(__file__), "dfdc_deepfake_challenge"))
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set
from training.zoo.classifiers import DeepFakeClassifier

# ----- App UI -----
st.set_page_config(page_title="DeepFake Generator & Classifier", layout="centered")
st.title("🔀 DeepFake Generator, Detector & AI Text Classifier")

# ----- Sidebar Mode Selection -----
mode = st.sidebar.radio("Select Mode", ["Generate Deepfake", "Classify Deepfake", "Detect AI Text", "Spam Detector","Multimodal Misinformation Detector","Metrics"])

# ----- Common Config -----
arcface_path = "arcface_model/arcface_checkpoint.tar"
crop_size = 224
name = "people"
temp_path = "./tmp_results"
os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs(temp_path, exist_ok=True)

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
report_dict = {
    'Class': ['True', 'Satire', 'False Conn', 'Imposter', 'Manipulated', 'Misleading'],
    'Precision': [0.89, 0.83, 0.78, 0.80, 0.88, 0.80],
    'Recall': [0.91, 0.64, 0.77, 0.41, 0.93, 0.80],
    'F1-Score': [0.90, 0.72, 0.77, 0.54, 0.90, 0.80],
    'Support': [1251, 124, 388, 39, 622, 75]
}
report_df = pd.DataFrame(report_dict)

# ----- DeepFake Video Classification Model -----
@st.cache_resource
def load_deepfake_model(weights_path):
    model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
    model.eval()
    return model.half()

# ----- AI Text Classifier Model -----
@st.cache_resource
def load_ai_text_model():
    tokenizer = AutoTokenizer.from_pretrained("./ai_detector")
    model_path = "./ai_detector"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model from .bin: {e}")
        model_weights = load_file(f"{model_path}/model.safetensors")
        model = AutoModelForSequenceClassification.from_pretrained(model_path, state_dict=model_weights)
    return tokenizer, model

# ----- Spam Detector Model -----
@st.cache_resource
def load_spam_detector_model():
    tokenizer = AutoTokenizer.from_pretrained("./spam_detector")
    model = AutoModelForSequenceClassification.from_pretrained("./spam_detector")
    return tokenizer, model

#MultiModal Ability
# Cache the model loading function
@st.cache_resource
def load_model():
    model = BERTResNetClassifier(num_classes=6)
    model.load_state_dict(torch.load("./image_text/models/model_epoch_316.pth", map_location="cpu"),strict=False)
    model.to(device)
    model.eval()
    return model
@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")

def get_bert_embedding(text, tokenizer, max_len=64):
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)

def predict(model, tokenizer, text, image_path):
    input_ids, attention_mask = get_bert_embedding(text, tokenizer)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor, input_ids, attention_mask)
        probs = F.softmax(output, dim=1).cpu().numpy().flatten()
        pred = torch.argmax(output, dim=1).item()

    return pred, probs, image

# ----- Mode 1: Generate Deepfake -----
if mode == "Generate Deepfake":
    st.header("🎭 SimSwap Deepfake Generator")
    uploaded_image = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"], key="gen_img")
    uploaded_video = st.file_uploader("Upload Target Video", type=["mp4", "avi", "mov"], key="gen_vid")

    if st.button("Generate Deepfake"):
        if uploaded_image and uploaded_video:
            # Create unique file names
            unique_id = uuid.uuid4().hex[:8]
            img_ext = os.path.splitext(uploaded_image.name)[-1]
            vid_ext = os.path.splitext(uploaded_video.name)[-1]

            img_filename = f"img_{unique_id}{img_ext}"
            vid_filename = f"vid_{unique_id}{vid_ext}"
            output_filename = f"deepfake_{unique_id}.mp4"

            img_path = os.path.join("input", img_filename)
            vid_path = os.path.join("input", vid_filename)
            output_path = os.path.join("output/", output_filename)

            # Save files to disk
            with open(img_path, "wb") as f:
                f.write(uploaded_image.read())
            with open(vid_path, "wb") as f:
                f.write(uploaded_video.read())

            # SimSwap command
            command = [
                "python", "test_video_swapsingle.py",
                "--isTrain", "false",
                "--crop_size", str(crop_size),
                "--name", name,
                "--Arc_path", arcface_path,
                "--pic_a_path", img_path,
                "--video_path", vid_path,
                "--output_path", output_path,
            ]

            # Progress bar for deepfake generation
            progress_bar = st.progress(0)
            with st.spinner("🛠️ Generating deepfake..."):
                # Run the subprocess and capture stdout and stderr
                process = subprocess.Popen(
                    command, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True, 
                    encoding='utf-8',
                    cwd="./SimSwap"
                )

                # Simulate progress based on output (you can adjust this based on your script's output)
                for stdout_line in iter(process.stdout.readline, ""):
                    st.text(stdout_line.strip())  # Display stdout in Streamlit
                    
                    # Simulate progress update (you may need to modify based on real stdout feedback)
                    if "progress" in stdout_line.lower():  # Replace this with actual condition based on stdout
                        progress_bar.progress(50)  # Update progress to 50% for demonstration (replace with actual calculation)
                
                # Handle stderr (error output)
                for stderr_line in iter(process.stderr.readline, ""):
                    st.text(stderr_line.strip())  # Display stderr in Streamlit

                process.stdout.close()
                process.stderr.close()
                process.wait()  # Wait for process to finish

            # Check if the deepfake generation was successful
            if os.path.exists(output_path):
                st.success("✅ Deepfake generated successfully!")
                st.video(output_path)
                st.markdown(f"📁 **Saved as:** `{output_filename}`")

                with open(output_path, "rb") as video_file:
                    st.download_button(
                        label="⬇️ Download Video",
                        data=video_file,
                        file_name=output_filename,
                        mime="video/mp4"
                    )
            else:
                st.error("❌ Error during deepfake generation. No output file found.")
        else:
            st.warning("⚠️ Please upload both a source image and a target video.")


# ----- Mode 2: Classify Deepfake -----
elif mode == "Classify Deepfake":
    st.header("🧠 Deepfake Video Detector")
    uploaded_file = st.file_uploader("Upload a video file (MP4)", type=["mp4"], key="clf_vid")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        st.video(video_path)

        if st.button("🔍 Predict Deepfake"):
            st.write("Analyzing the video, please wait...")

            frames_per_video = 32
            input_size = 380
            model_paths = ["dfdc_deepfake_challenge/weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36"]
            models = [load_deepfake_model(p) for p in model_paths]

            video_reader = VideoReader()
            video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
            face_extractor = FaceExtractor(video_read_fn)
            strategy = confident_strategy

            predictions_dict, frames_info = predict_on_video_set(
                face_extractor=face_extractor,
                input_size=input_size,
                models=models,
                strategy=strategy,
                frames_per_video=frames_per_video,
                videos=[os.path.basename(video_path)],
                num_workers=2,
                test_dir=os.path.dirname(video_path),
            )

            video_name = os.path.basename(video_path)
            result = predictions_dict[video_name]
            frame_data = frames_info[video_name]

            st.subheader("Prediction Result")

            labels = ['REAL', 'FAKE']
            mean_probability = np.mean(result)  # Mean probability for the video being fake
            confidence_scores = [1 - mean_probability, mean_probability]

           
            df_conf = pd.DataFrame({
                "Label": labels,
                "Confidence": confidence_scores
            })

            fig = px.pie(
                df_conf,
                values="Confidence",
                names="Label",
                color="Label",
                color_discrete_map={"REAL": "green", "FAKE": "red"},
                title="Prediction Confidence",
                hole=0.3
            )
            fig.update_traces(textinfo='percent+label', pull=[0, 0.1])
            st.plotly_chart(fig, use_container_width=True)

            if mean_probability  > 0.5:
                st.markdown(f"<h3 style='color:red;'>FAKE ❌ (Confidence: {mean_probability:.2f})</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:green;'>REAL ✅ (Confidence: {1 - mean_probability:.2f})</h3>", unsafe_allow_html=True)

            # Optionally show top N suspicious frames in a left-to-right manner
            st.subheader("Suspicious Frames")
            frame_data_sorted = sorted(frame_data, key=lambda x: x[0], reverse=True)
            top_n = len(frame_data_sorted)

            # Create columns dynamically based on the number of images
            num_columns = 5  # You can adjust this number to control the number of columns
            columns = st.columns(num_columns)

            for i in range(top_n):
                prob, img = frame_data_sorted[i]
                col = columns[i % num_columns]  # Rotate through the columns
                col.image(img, caption=f"Fake Score: {prob:.2f}", use_column_width=True)

            os.remove(video_path)


# ----- Mode 3: Detect AI Text -----
elif mode == "Detect AI Text":
    st.header("🧠 AI Text Detector")
    st.markdown("Analyze a piece of text to check if it was written by a human or AI.")

    input_text = st.text_area("✍️ Enter text to analyze:", height=150)

    if st.button("Detect"):
        if input_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            tokenizer, model = load_ai_text_model()
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            pred_label = torch.argmax(predictions).item()
            confidence_scores = predictions[0].tolist()

            labels = ["Human-written", "AI-generated"]
            confidence = confidence_scores[pred_label]

            st.markdown(f"### 🏷️ **Result:** {labels[pred_label]}")
            st.markdown(f"**Confidence:** `{confidence:.2%}`")

            df = pd.DataFrame({
                "Label": labels,
                "Confidence": confidence_scores
            })

            fig = px.bar(df, x="Label", y="Confidence", color="Label", 
                         range_y=[0, 1],
                         text=df["Confidence"].map(lambda x: f"{x:.2%}"),
                         color_discrete_sequence=["#2ecc71", "#e74c3c"])
            fig.update_traces(textposition='outside')
            fig.update_layout(title="Prediction Confidence", yaxis_title="Confidence", xaxis_title="Label")

            st.plotly_chart(fig)

            if pred_label == 0:
                st.success("✅ The model thinks this text is **likely written by a human**.")
            else:
                st.error("🤖 This text appears to be **AI-generated** based on the model's prediction.")

# ----- Mode 4: Spam Detector -----
elif mode == "Spam Detector":
    st.title("Spam Detector")

    input_text = st.text_area("Enter text to analyze:", height=150)

    if st.button("Detect"):
        if input_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            tokenizer, model = load_spam_detector_model()
            inputs = tokenizer([input_text], return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            label_map = {0: "Not Spam", 1: "Spam"}

            pred_label = label_map[predictions.item()]
            confidence = probabilities[0][predictions.item()].item()

            st.subheader("Result")
            st.write(f"**Label:** {pred_label} (Confidence: {confidence:.2f})")

            if pred_label == "Spam":
                st.error("Warning: This message is detected as Spam! Avoid clicking on suspicious links or providing sensitive information.")
            else:
                st.success("This message is not detected as spam. It's safe!")

            fig = go.Figure()
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

            st.plotly_chart(fig)

elif mode == "Multimodal Misinformation Detector":
    st.title("🧠 Multimodal Misinformation News Detector")
    st.markdown("Predict the type of news post based on **image** and **text** using a fine-tuned BERT + ResNet model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()
    tokenizer = load_tokenizer()

    label_map = {
        0: "True Content",
        1: "Satire / Parody",
        2: "False Connection",
        3: "Imposter Content",
        4: "Manipulated Content",
        5: "Misleading Content"
    }

    category_descriptions = {
        "True Content": "This refers to information that is **factually correct and verified** through reliable sources. The headline, body, and accompanying visuals (if any) are aligned and do not mislead the reader in any way. It is authentic journalism or accurate content, free from distortion or manipulation.",

        "Satire / Parody": "This content is intentionally created for **humor or entertainment**, often using exaggeration or irony to comment on current events, politics, or society. While it may resemble real news, it does not aim to deceive the audience, and the source is often known for its satirical nature (e.g., The Onion, Babylon Bee).",

        "False Connection": "This involves **misleading associations** between the headline, images, or captions and the actual content. For example, a dramatic headline or viral image might be paired with unrelated or less impactful article text, tricking the viewer into believing a false narrative without overtly fabricating facts.",

        "Imposter Content": "This content **mimics the branding or appearance of credible sources** like news organizations, government bodies, or public figures. It uses fake logos, names, or impersonated social media profiles to trick users into believing the information comes from a trustworthy entity.",

        "Manipulated Content": "This includes **photos, videos, or audio that have been edited** or altered to change their original meaning or context. Deepfakes, Photoshop edits, and spliced audio fall into this category. The changes are usually subtle but meant to deceive or provoke a specific reaction.",

        "Misleading Content": "This content **uses factual information in a deceptive way**, often by omitting context, cherry-picking data, or misrepresenting sources. It can exaggerate or oversimplify complex issues to promote a particular agenda or bias while technically staying within the bounds of truth."
    }

    colors = ['#88B04B', '#FF6F61', '#6B5B95', '#F7CAC9', '#92A8D1', '#955251']

    uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    input_text = st.text_area("📝 Enter the post's text/title")

    if st.button("🚀 Predict"):
        if uploaded_image and input_text:
            with st.spinner("Running inference..."):
                pred_class_idx, probs, display_image = predict(model, tokenizer, input_text, uploaded_image)

            pred_label = label_map[pred_class_idx]
            st.image(display_image, caption="Uploaded Image", use_column_width=True)

            # Plotly bar chart
            fig_bar = px.bar(
                x=list(label_map.values()),
                y=probs,
                labels={'x': 'Category', 'y': 'Probability'},
                title="Prediction Probabilities by Category",
                color=list(label_map.values()),
                color_discrete_sequence=colors
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Plotly pie chart
            fig_pie = px.pie(
                values=probs,
                names=list(label_map.values()),
                title="Class Probability Distribution",
                color=list(label_map.values()),
                color_discrete_sequence=colors
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Predicted Label
            st.markdown(f"### ✅ **Predicted Class:** `{pred_label}`")
            st.info(category_descriptions[pred_label])

            # Interactive Explanation
            category_colors = {
                "True Content": "#d4edda",        # Light green
                "Satire / Parody": "#fff3cd",     # Light yellow
                "False Connection": "#d1ecf1",    # Light cyan
                "Imposter Content": "#f8d7da",    # Light red
                "Manipulated Content": "#e2e3e5", # Light gray
                "Misleading Content": "#fefefe"   # Near white
            }

            with st.expander("📘 What does each category mean?"):
                for key in label_map.values():
                    st.markdown(f"""
                    <div style="margin-bottom: 20px; padding: 15px; border-radius: 10px; background-color: {category_colors[key]}; border: 1px solid #ccc;">
                        <h4 style="color:#333;">🧩 {key}</h4>
                        <p style="margin-top: -10px; color: #555;">{category_descriptions[key]}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Please provide both an image and a text input to proceed.")

elif mode=="Metrics":
    # --- App Title ---
    st.title("📊 Model Performance Dashboard")

    # --- Model Selector ---
    selected_model = st.selectbox("Choose a model to view its metrics:", list(model_metrics.keys()))
    metrics = model_metrics[selected_model]
    df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']).dropna()

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

# ----- Footer -----
st.markdown("---")
st.markdown(
    """
    <style>
    div[data-baseweb="select"] > div {
        cursor: pointer;
    }
    </style>
    <div style="text-align: center; font-size: 0.9em; color: grey;">
        Built with Neural Networks ❤️ <br>
        Powered by <strong>SimSwap</strong>, <strong>DFDCP Classifier</strong>, and <strong>HuggingFace Transformers</strong>
    </div>
    """,
    unsafe_allow_html=True
)
