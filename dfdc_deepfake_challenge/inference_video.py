import os
from glob import glob
from classifier import DeepFakeClassifier
import torch
import cv2
from torchvision import transforms
from facenet_pytorch import MTCNN
import numpy as np

# Load model
model_path = 'weights/final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepFakeClassifier(encoder='tf_efficientnet_b7_ns', num_classes=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# Face detector
mtcnn = MTCNN(select_largest=True, device=device)

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# Load video
video_path = 'test_videos/fake21.mp4'
cap = cv2.VideoCapture(video_path)
frames = []
count = 0

while True:
    ret, frame = cap.read()
    if not ret or count > 100:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(frame_rgb)
    if face is not None:
        face = preprocess(face.permute(1, 2, 0).byte().numpy())
        frames.append(face)
    count += 1

cap.release()

if frames:
    batch = torch.stack(frames).to(device)
    with torch.no_grad():
        preds = torch.sigmoid(model(batch)).squeeze().cpu().numpy()
        avg_score = float(np.mean(preds))
        print(f"\nPrediction: {'FAKE' if avg_score > 0.5 else 'REAL'} (score={avg_score:.4f})")
else:
    print("No face detected in video.")
