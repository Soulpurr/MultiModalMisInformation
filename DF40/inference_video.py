import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import timm
import torch.nn as nn
from torchvision import transforms
from facenet_pytorch import MTCNN  # face detector

# Load Xception model
def get_model(weights_path='xception.pth'):
    model = timm.create_model('xception', pretrained=False, num_classes=2)
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

# Face detector (only once)
mtcnn = MTCNN(select_largest=True, post_process=False, device='cuda' if torch.cuda.is_available() else 'cpu')

# Frame preprocessor
def preprocess_frame(face):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    return preprocess(face).unsqueeze(0)

# Inference function
def run_inference(video_path, model, device='cpu'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fake_scores = []
    processed_frames = 0

    with torch.no_grad():
        for _ in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = mtcnn(rgb_frame)

            if face is not None:
                face = face.permute(1, 2, 0).cpu().numpy()  # HWC
                face = (np.clip(face, 0, 1) * 255).astype(np.uint8)  # Convert to uint8
                input_tensor = preprocess_frame(face).to(device)
                output = model(input_tensor)
                prob = torch.softmax(output, dim=1)[0, 1].item()
                fake_scores.append(prob)
                processed_frames += 1

    cap.release()

    if fake_scores:
        avg_fake_score = np.mean(fake_scores)
        print(f"\nPrediction: {'FAKE' if avg_fake_score > 0.5 else 'REAL'} ({avg_fake_score:.4f})")
    else:
        print("No faces detected or no frames processed.")

# Main
if __name__ == "__main__":
    video_path = "./video_input/fake21.mp4"         # Update this path
    weights_path = "./weights/xception.pth"         # Update this path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(weights_path).to(device)
    run_inference(video_path, model, device)
