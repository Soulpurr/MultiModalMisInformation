import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import yaml
import os
import sys

# Add project root to path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
print("✅ Model loaded.")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load video
video_path = "video_input/res21.mp4"
cap = cv2.VideoCapture(video_path)

preds = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 10 == 0:  # Sample every 10th frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model({"image": img_tensor}, inference=True)
            prob = torch.sigmoid(output["prob"]).item()
            preds.append(prob)

    frame_count += 1

cap.release()

# Report result
if preds:
    avg_score = np.mean(preds)
    print(f"\n🎥 Video: {video_path}")
    print(f"🔍 Deepfake Score (avg of sampled frames): {avg_score:.4f}")
    print("🔴 Likely Fake" if avg_score > 0.5 else "🟢 Likely Real")
else:
    print("⚠️ No frames were processed.")
