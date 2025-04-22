# Import necessary libraries
import streamlit as st
import subprocess
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

# ----- Local Imports -----
sys.path.append(os.path.join(os.path.dirname(__file__), "dfdc_deepfake_challenge"))
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set
from training.zoo.classifiers import DeepFakeClassifier

# ----- App UI -----
st.set_page_config(page_title="DeepFake Generator & Classifier", layout="centered")
st.title("🔀 DeepFake Generator, Detector & AI Text Classifier")

# ----- Sidebar Mode Selection -----
mode = st.sidebar.radio("Select Mode", ["Generate Deepfake", "Classify Deepfake", "Detect AI Text", "Spam Detector"])

# ----- Common Config -----
arcface_path = "SimSwap/arcface_model/arcface_checkpoint.tar"
crop_size = 224
name = "people"
temp_path = "./tmp_results"
os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs(temp_path, exist_ok=True)

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

# ----- Mode 1: Generate Deepfake -----
if mode == "Generate Deepfake":
    st.header("🎭 SimSwap Deepfake Generator")
    uploaded_image = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"], key="gen_img")
    uploaded_video = st.file_uploader("Upload Target Video", type=["mp4", "avi", "mov"], key="gen_vid")

    if st.button("🚀 Begin Deepfake Generation"):
        if uploaded_image and uploaded_video:
            unique_id = uuid.uuid4().hex[:8]
            img_ext = os.path.splitext(uploaded_image.name)[-1]
            vid_ext = os.path.splitext(uploaded_video.name)[-1]

            img_path = os.path.join("input", f"img_{unique_id}{img_ext}")
            vid_path = os.path.join("input", f"vid_{unique_id}{vid_ext}")
            output_path = os.path.join("output", f"deepfake_{unique_id}.mp4")

            with open(img_path, "wb") as f:
                f.write(uploaded_image.read())
            with open(vid_path, "wb") as f:
                f.write(uploaded_video.read())

            command = [
                "python", "SimSwap/test_video_swapsingle.py",
                "--isTrain", "false",
                "--crop_size", str(crop_size),
                "--name", name,
                "--Arc_path", arcface_path,
                "--pic_a_path", img_path,
                "--video_path", vid_path,
                "--output_path", output_path,
            ]

            with st.spinner("🛠️ Generating deepfake..."):
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')

            if result.returncode == 0:
                st.success("✅ Deepfake generated successfully!")
                st.video(output_path)
                with open(output_path, "rb") as video_file:
                    st.download_button("⬇️ Download Video", data=video_file, file_name=os.path.basename(output_path), mime="video/mp4")
            else:
                st.error("❌ Error during deepfake generation.")
                st.code(result.stderr)
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

            predictions = predict_on_video_set(
                face_extractor=face_extractor,
                input_size=input_size,
                models=models,
                strategy=strategy,
                frames_per_video=frames_per_video,
                videos=[os.path.basename(video_path)],
                num_workers=2,
                test_dir=os.path.dirname(video_path),
            )

            result = predictions[0]
            st.subheader("Prediction Result")

            labels = ['REAL', 'FAKE']
            confidence_scores = [1 - result, result]

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

            if result > 0.5:
                st.markdown(f"<h3 style='color:red;'>FAKE ❌ (Confidence: {result:.2f})</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:green;'>REAL ✅ (Confidence: {1 - result:.2f})</h3>", unsafe_allow_html=True)

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

# ----- Footer -----
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 0.9em; color: grey;">
        Built with ❤️ by Abhi<br>
        Powered by <strong>SimSwap</strong>, <strong>DFDCP Classifier</strong>, and <strong>HuggingFace Transformers</strong>
    </div>
    """,
    unsafe_allow_html=True
)
