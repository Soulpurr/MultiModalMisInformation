# Multimodal Misinformation Classifier

A powerful pipeline for detecting misinformation through deepfake detection, identity-swapping, and multimodal (image + text) classification. This system integrates deep learning models trained on large-scale datasets like DFDC and Fakeddit to detect fake content across modalities.

---

## 📁 Folder Overview

This repo includes multiple specialized modules, some of which are excluded from Git due to large file sizes. You will need to manually download model weights into the specified directories.

| Directory | Description |
|----------|-------------|
| `DF40/` | Deepfake detection project, includes pretrained SimSwap and EfficientNet models |
| `dfdc_deepfake_challenge/weights/` | Pretrained DFDC model weights (.pth files) |
| `dfdc_deepfake_challenge/test_videos/` | Sample videos from DFDC dataset |
| `image_text/models/` | Image-text fusion model weights |
| `image_text/images/` | Sample images for inference |
| `ai_detector/model.safetensors` | EfficientNet weights in `.safetensors` format |
| `SimSwap/checkpoints/` | Pretrained SimSwap face-swapping weights |
| `SimSwap/arcface_model/` | ArcFace face recognition model used in SimSwap |

> **Note:** These folders are ignored in `.gitignore` to avoid tracking large files. You must download them manually.

---

## 🧠 Module Descriptions

### 🔄 SimSwap (Identity Swapping)

- **Location:** `SimSwap/`
- Face swapping network used for deepfake creation and adversarial testing
- Uses ArcFace embeddings for identity preservation
- Based on the paper: [SimSwap: An Efficient Framework for High Fidelity Face Swapping](https://arxiv.org/abs/2106.06340)  
  GitHub: [https://github.com/neuralchen/SimSwap](https://github.com/neuralchen/SimSwap)

### 🎭 DFDC Deepfake Detection (EfficientNet)

- **Location:** `dfdc_deepfake_challenge/` and `ai_detector/`
- EfficientNet B4 used as the backbone
- Trained on the [DeepFake Detection Challenge (DFDC)](https://www.kaggle.com/competitions/deepfake-detection-challenge/)
- Inspired by the top-winning Kaggle solution by NTechLab (1st place)  
  [Winning solution discussion](https://www.kaggle.com/competitions/deepfake-detection-challenge/discussion/164984)

### 🖼️📝 Multimodal Classifier (Image + Text)

- **Location:** `image_text/`
- Combines image and text using **early fusion**
- Trained on the [Fakeddit dataset](https://github.com/williamleif/fakeddit)
- Supports binary or multi-class misinformation classification

---

## 🛠️ Setup & Usage

Follow these steps to set up the environment and run the application.

---

### 1️⃣ Install Dependencies

Make sure you have Python ≥ 3.8 installed. Then install all required packages:
  pip install -r requirements.txt

### 2️⃣ Download Model Weights
Since large model files are ignored in the repository (see .gitignore), you'll need to manually download and place them in the appropriate directories:

Folder	What to Download
DF40/	DeepfakeBench + SimSwap weights (e.g., *.pth)
dfdc_deepfake_challenge/weights/	DFDC models (e.g., xception.pth)
dfdc_deepfake_challenge/test_videos/	Sample input videos
image_text/models/	Trained Fakeddit model
image_text/images/	Test images
ai_detector/	model.safetensors file
SimSwap/checkpoints/	SimSwap face-swapping model
SimSwap/arcface_model/	ArcFace identity embedding model

✅ Keep a .gitkeep in each folder so Git tracks the empty directory structure.

### 3️⃣ Run the Streamlit App
Once weights are in place, launch the UI using:
 streamlit run app.py
This opens a web interface for uploading images, videos, or text to classify misinformation and detect deepfakes.

### 🧪 Example Use Cases
Upload a video clip to test for deepfakes

Upload an image with a caption to test misinformation prediction

Combine both for robust multimodal analysis

### 📫 Questions?
Feel free to open an issue or reach out for help getting the models running locally.

### 📜 License
MIT License

You are free to use, modify, and distribute this project with proper attribution.

### 📚 References
  SimSwap: SimSwap: An Efficient Framework for High Fidelity Face Swapping, Chen et al., 2021
  GitHub: https://github.com/neuralchen/SimSwap
  
  DFDC Challenge:
  DeepFake Detection Challenge on Kaggle
  1st Place Solution Discussion by NTechLab
  
  Fakeddit Dataset:
  Fakeddit: A New Multimodal Benchmark Dataset for Fake News Detection
