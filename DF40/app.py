import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from tempfile import NamedTemporaryFile
import timm
from torchvision import transforms
from tqdm import tqdm

# Load model function
@st.cache_resource
def load_model(weights_path='weights/core_best.pth'):
    model = timm.create_model('xception', pretrained=False, num_classes=2)
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

# Frame preprocessing
def preprocess_frame(frame):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    return preprocess(frame).unsqueeze(0)

# Inference function
def run_inference(video_path, model, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fake_scores = []

    with torch.no_grad():
        for _ in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess_frame(frame).to(device)
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)[0, 1].item()
            fake_scores.append(prob)

    cap.release()

    if fake_scores:
        avg_fake_score = np.mean(fake_scores)
        return avg_fake_score
    else:
        return None

# Streamlit App UI
st.title("🎭 DeepFake Video Detection - Xception")

uploaded_video = st.file_uploader("Upload a video (.mp4)", type=["mp4"])
device_option = st.radio("Select Device:", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"])

if uploaded_video is not None:
    with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    st.video(temp_video_path)

    if st.button("Run Detection"):
        with st.spinner("Loading model..."):
            model = load_model('weights/core_best.pth').to(device_option)

        st.write("Processing video...")
        avg_score = run_inference(temp_video_path, model, device_option)

        if avg_score is not None:
            label = "FAKE" if avg_score > 0.5 else "REAL"
            st.success(f"Prediction: **{label}** ({avg_score:.4f})")
        else:
            st.warning("No frames processed.")
