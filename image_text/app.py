import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
import plotly.express as px
import torch.nn.functional as F

# ===========================
# Model Definition
# ===========================
class BERTResNetClassifier(torch.nn.Module):
    def __init__(self, num_classes=6):
        super(BERTResNetClassifier, self).__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        from transformers import BertModel
        self.image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.fc_image = torch.nn.Linear(1000, num_classes)
        self.drop = torch.nn.Dropout(0.3)
        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.fc_text = torch.nn.Linear(self.text_model.config.hidden_size, num_classes)

    def forward(self, image, input_ids, attention_mask):
        x_img = self.image_model(image)
        x_img = self.drop(x_img)
        x_img = self.fc_image(x_img)

        x_text = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0][:, 0, :]
        x_text = self.drop(x_text)
        x_text = self.fc_text(x_text)

        return torch.max(x_text, x_img)

# ===========================
# Utilities
# ===========================
@st.cache_resource
def load_model():
    model = BERTResNetClassifier()
    model.load_state_dict(torch.load("./models/modellast.pth", map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")

def get_bert_embedding(text, tokenizer, max_len=64):
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)

def predict(model, tokenizer, text, image_path):
    input_ids, attention_mask = get_bert_embedding(text, tokenizer)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor, input_ids, attention_mask)
        probs = F.softmax(output, dim=1).cpu().numpy().flatten()
        pred = torch.argmax(output, dim=1).item()

    return pred, probs, image

# ===========================
# Streamlit App
# ===========================
st.set_page_config(page_title="Multimodal Fake News Detector", layout="centered")
st.title("🧠 Multimodal Fake News Classifier")
st.markdown("Predict the type of news post based on **image** and **text** using a fine-tuned BERT + ResNet model.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model()
tokenizer = load_tokenizer()

label_map = {
    0: "True Content",
    1: "Satire / Parody",
    2: "False Connection",
    3: "Imposter Content",
    4: "Manipulated Content",
    5: "Misleading Content"
}

category_descriptions = {
    "True Content": "This refers to information that is **factually correct and verified** through reliable sources. The headline, body, and accompanying visuals (if any) are aligned and do not mislead the reader in any way. It is authentic journalism or accurate content, free from distortion or manipulation.",

    "Satire / Parody": "This content is intentionally created for **humor or entertainment**, often using exaggeration or irony to comment on current events, politics, or society. While it may resemble real news, it does not aim to deceive the audience, and the source is often known for its satirical nature (e.g., The Onion, Babylon Bee).",

    "False Connection": "This involves **misleading associations** between the headline, images, or captions and the actual content. For example, a dramatic headline or viral image might be paired with unrelated or less impactful article text, tricking the viewer into believing a false narrative without overtly fabricating facts.",

    "Imposter Content": "This content **mimics the branding or appearance of credible sources** like news organizations, government bodies, or public figures. It uses fake logos, names, or impersonated social media profiles to trick users into believing the information comes from a trustworthy entity.",

    "Manipulated Content": "This includes **photos, videos, or audio that have been edited** or altered to change their original meaning or context. Deepfakes, Photoshop edits, and spliced audio fall into this category. The changes are usually subtle but meant to deceive or provoke a specific reaction.",

    "Misleading Content": "This content **uses factual information in a deceptive way**, often by omitting context, cherry-picking data, or misrepresenting sources. It can exaggerate or oversimplify complex issues to promote a particular agenda or bias while technically staying within the bounds of truth."
}

colors = ['#88B04B', '#FF6F61', '#6B5B95', '#F7CAC9', '#92A8D1', '#955251']

uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
input_text = st.text_area("📝 Enter the post's text/title")

if st.button("🚀 Predict"):
    if uploaded_image and input_text:
        with st.spinner("Running inference..."):
            pred_class_idx, probs, display_image = predict(model, tokenizer, input_text, uploaded_image)

        pred_label = label_map[pred_class_idx]
        st.image(display_image, caption="Uploaded Image", use_column_width=True)

        # Plotly bar chart
        fig_bar = px.bar(
            x=list(label_map.values()),
            y=probs,
            labels={'x': 'Category', 'y': 'Probability'},
            title="Prediction Probabilities by Category",
            color=list(label_map.values()),
            color_discrete_sequence=colors
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Plotly pie chart
        fig_pie = px.pie(
            values=probs,
            names=list(label_map.values()),
            title="Class Probability Distribution",
            color=list(label_map.values()),
            color_discrete_sequence=colors
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Predicted Label
        st.markdown(f"### ✅ **Predicted Class:** `{pred_label}`")
        st.info(category_descriptions[pred_label])

        # Interactive Explanation
        category_colors = {
            "True Content": "#d4edda",        # Light green
            "Satire / Parody": "#fff3cd",     # Light yellow
            "False Connection": "#d1ecf1",    # Light cyan
            "Imposter Content": "#f8d7da",    # Light red
            "Manipulated Content": "#e2e3e5", # Light gray
            "Misleading Content": "#fefefe"   # Near white
        }

        with st.expander("📘 What does each category mean?"):
            for key in label_map.values():
                st.markdown(f"""
                <div style="margin-bottom: 20px; padding: 15px; border-radius: 10px; background-color: {category_colors[key]}; border: 1px solid #ccc;">
                    <h4 style="color:#333;">🧩 {key}</h4>
                    <p style="margin-top: -10px; color: #555;">{category_descriptions[key]}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Please provide both an image and a text input to proceed.")
