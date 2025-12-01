import streamlit as st
from PIL import Image
import torch
from models import get_mobilenet, SimpleViT
from torchvision import transforms
import json
import os

st.title('Face Recognition Demo')
st.write('Upload an image and get predicted identity.')

uploaded = st.file_uploader('Image', type=['jpg','jpeg','png'])

arch = st.selectbox('Model', ['cnn', 'vit'])
model_path = st.text_input('Model path', 'models/cnn_best.pth')

# otomatis mencari JSON mapping
def load_class_mapping(model_path):
    json_path = model_path.replace(".pth", "_classes.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        return idx_to_class
    return None

idx_to_class = load_class_mapping(model_path)

if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Input Image', use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    x = transform(img).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tentukan num_classes berdasarkan mapping (lebih aman)
    if idx_to_class is None:
        st.error("Mapping class tidak ditemukan. Pastikan file *_classes.json ada.")
        st.stop()

    num_classes = len(idx_to_class)

    # Load model
    if arch == 'cnn':
        model = get_mobilenet(num_classes=num_classes, pretrained=False)
    else:
        model = SimpleViT(num_classes=num_classes, pretrained=False)

    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()

        with torch.no_grad():
            logits = model(x.to(device))
            pred_idx = logits.argmax(dim=1).item()

        # Ambil nama
        if pred_idx in idx_to_class:
            class_name = idx_to_class[pred_idx]
        else:
            class_name = "Unknown"

        st.success(f"Predicted ID: {class_name} (index = {pred_idx})")

    except Exception as e:
        st.error('Error loading model: ' + str(e))
