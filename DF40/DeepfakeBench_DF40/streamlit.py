import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import yaml
from training.detectors import DETECTOR

# Load config
with open("training/config/detector/xception.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model from DETECTOR registry
model_class = DETECTOR[config["model_name"]]
model = model_class(config).to(device)
model.eval()

# Load weights
weights_path = "training/df40_weights/train_on_fs_matrix/simswap_ff.pth"
state_dict = torch.load(weights_path, map_location=device)
if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

# Adjust key names if needed
new_state = {}
for k, v in state_dict.items():
    k = k.replace("module.", "")
    k = k.replace("base_model.", "backbone.")
    k = k.replace("classifier.", "head.")
    new_state[k] = v

model.load_state_dict(new_state, strict=True)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Streamlit UI
st.title("Deepfake Detection")
st.write("Upload a video to check if it's real or fake.")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    video_bytes = uploaded_video.read()

    # Save video file
    file_path = "uploaded_video.mp4"
    with open(file_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(file_path)
    
    preds = []
    frame_count = 0
    frame_threshold = 0.7  # Frame-level probability threshold
    frame_votes = 0  # Counter for voting mechanism
    total_frames = 0  # Total frames processed

    # Initialize face detector (Haar Cascade or other model)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Face detection: only process frames with faces detected
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:  # Process frames with faces
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model({"image": img_tensor}, inference=True)
                prob = torch.sigmoid(output["prob"]).item()
                preds.append(prob)

                if prob > frame_threshold:
                    frame_votes += 1
                total_frames += 1

        frame_count += 1

    cap.release()

    # Calculate results
    if preds:
        avg_score = np.mean(preds)
        if total_frames > 0:
            # Voting mechanism (Majority rule based on frame probabilities)
            vote_percentage = (frame_votes / total_frames) * 100

            st.write(f"🎥 Video uploaded: {uploaded_video.name}")
            st.write(f"🔍 Deepfake Score (avg of sampled frames): {avg_score:.4f}")
            st.write(f"⚖️ Frame Votes: {vote_percentage:.2f}% frames flagged as fake")

            if vote_percentage > 60:  # Threshold for classification based on voting
                st.write("🔴 Likely Fake")
            else:
                st.write("🟢 Likely Real")
        else:
            st.write("⚠️ No faces detected in video frames.")
    else:
        st.write("⚠️ No frames were processed.")
