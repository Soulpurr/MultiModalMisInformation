import streamlit as st
import tempfile
import os
import torch
import pandas as pd
import re
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set
from training.zoo.classifiers import DeepFakeClassifier

# Load model(s) once
@st.cache_resource
def load_model(weights_path):
    model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
    model.eval()
    return model.half()

st.title("🎭 Deepfake Video Detector")

uploaded_file = st.file_uploader("Upload a video file (MP4)", type=["mp4"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("🔍 Predict"):
        st.write("Analyzing the video, please wait...")

        frames_per_video = 32
        input_size = 380

        model_paths = ["weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36"]
        models = [load_model(p) for p in model_paths]

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

        if result > 0.5:
            st.markdown(f"<h3 style='color:red;'>FAKE ❌ (Confidence: {result:.2f})</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:green;'>REAL ✅ (Confidence: {1 - result:.2f})</h3>", unsafe_allow_html=True)

        os.remove(video_path)
