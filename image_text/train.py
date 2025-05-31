import os
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from transformers import BertTokenizer, BertModel
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

# ==================== Dataset ====================
class FakedditDataset(Dataset):
    def __init__(self, df, tokenizer, image_transform, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['clean_title']
        image_path = row['image']
        label = row['label']

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        # Tokenize text
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()

        return image, input_ids, attention_mask, label


# ==================== Model ====================
class BERTResNetClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(BERTResNetClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        self.text_proj = nn.Linear(self.bert.config.hidden_size, 256)
        self.img_proj = nn.Linear(2048, 256)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, text_input_ids, text_attention_mask):
        text_outputs = self.bert(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_feat = self.text_proj(text_outputs.last_hidden_state[:, 0, :])

        img_feat = self.resnet(image).squeeze(-1).squeeze(-1)
        img_feat = self.img_proj(img_feat)

        combined = torch.cat((text_feat, img_feat), dim=1)
        return self.classifier(combined)


# ==================== Training ====================
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for image, input_ids, attention_mask, label in tqdm(dataloader):
        image = image.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image, input_ids, attention_mask)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


# ==================== Evaluation ====================
def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for image, input_ids, attention_mask, label in tqdm(dataloader):
            image = image.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(image, input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(label.numpy())

    print(classification_report(y_true, y_pred))


# ==================== Main ====================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv_path)
    df = df.dropna(subset=['clean_title', 'image', 'label'])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    train_dataset = FakedditDataset(train_df, tokenizer, transform)
    val_dataset = FakedditDataset(val_df, tokenizer, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = BERTResNetClassifier(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Train loss: {train_loss:.4f}")
        evaluate_model(model, val_loader, device)

    torch.save(model.state_dict(), args.output_model_path)
    print(f"Model saved to {args.output_model_path}")


# ==================== CLI ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file with clean_title, image, label')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--output_model_path', type=str, default='multimodal_model.pth')

    args = parser.parse_args()
    main(args)
