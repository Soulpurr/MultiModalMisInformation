import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as f
import torch.optim as optim

import torchvision
from torchvision import transforms
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim.lr_scheduler as lr_scheduler

import random
from PIL import Image
import matplotlib.pyplot as plt

import torch
from transformers import BertModel, BertTokenizer

import torch

from PIL import Image
import pandas as pd


class BERTResNetClassifier(nn.Module):
    def __init__(self, num_classes=6):

        super(BERTResNetClassifier, self).__init__()

        self.num_classes = num_classes

        # Image processing (ResNet)
        self.image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Image processing (Fully Connected Layer)
        self.fc_image = nn.Linear(in_features=1000, out_features=num_classes, bias=True)

        # Dropout layer
        self.drop = nn.Dropout(p=0.3)

        # Text processing (using the 768-dimensional BERT arrays)
        self.text_model = BertModel.from_pretrained("bert-base-uncased")

        # Text processing (Fully Connected Layer)
        self.fc_text = nn.Linear(in_features=self.text_model.config.hidden_size, out_features=num_classes, bias=True)

        # Fusion and classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, text_input_ids, text_attention_mask,):
        # Image branch
        x_img = self.image_model(image)
        x_img = self.drop(x_img)
        x_img = self.fc_image(x_img)

        # Text branch
        x_text_last_hidden_states = self.text_model(
            input_ids = text_input_ids,
            attention_mask = text_attention_mask,
            return_dict=False
        )
        x_text_pooled_output = x_text_last_hidden_states[0][:, 0, :]
        x_text = self.drop(x_text_pooled_output)
        x_text = self.fc_text(x_text_pooled_output)

        # Fusion and max merge
        x = torch.max(x_text, x_img)

        # Classification
        #x = self.softmax(x) #-> already applied in crossentropy loss

        return x




# ========================
# Load model
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your model class here if not already defined
# from your_model_file import MultimodalModel

model = BERTResNetClassifier()  # Replace with your model class
model.load_state_dict(torch.load("./models/modellast.pth", map_location=device),strict=False)
model.to(device)
model.eval()



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_bert_embedding(text, max_len=64):
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
    return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)


import torch
from torchvision import transforms
from PIL import Image

def predict_single(model, text, image_path, label_map=None):
    model.eval()
    
    # Preprocess text
    input_ids, attention_mask = get_bert_embedding(text)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    # Preprocess image
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image, input_ids, attention_mask)
        pred_class = torch.argmax(output, dim=1).item()
    
    if label_map:
        return label_map[pred_class]  # e.g., label_map = {0: "True", 1: "Fake", ...}
    return pred_class

sample_text = "americas best dance sloth"
sample_image_path = "./images/imop.jpg"  # Update path as needed

predicted_label = predict_single(model, sample_text, sample_image_path)
print("Predicted label:", predicted_label)