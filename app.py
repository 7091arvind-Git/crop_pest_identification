import os
import json
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import streamlit as st

# ---------------- CONFIG ----------------
MODEL_PATH = "saved_models/pest_model.pth"
MAP_PATH = "meta/class_indices.json"

# Demo image from your session (used only if present)
DEMO_IMAGE_PATH = "/mnt/data/cf321941-e6a5-4ce2-a40b-dba98621ec6c.png"


# ---------------- LOAD MODEL + MAPPING ----------------
@st.cache_resource
def load_model():
    # load mapping
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        class_map = json.load(f)
    inv_map = {int(k): v for k, v in class_map.items()}
    num_classes = len(inv_map)

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    return model, inv_map, device


def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])
    return transform(img).unsqueeze(0)


def predict(model, device, img, inv_map, topk=3):
    x = preprocess(img).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
    idxs = probs.argsort()[-topk:][::-1]
    results = [(inv_map[i], probs[i]) for i in idxs]
    return results


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Crop Pest Identification", layout="centered")

st.title("🌾 Crop Pest Identification")
st.write("Upload a leaf/crop image to identify the pest using a trained MobileNetV2 model.")


# Load model
with st.spinner("Loading model..."):
    model, inv_map, device = load_model()


# Upload section
uploaded = st.file_uploader("Upload image (jpg/png)", type=["jpg", "jpeg", "png"])

# Demo image button
use_demo = False
if uploaded is None:
    if os.path.isfile(DEMO_IMAGE_PATH):
        if st.button("Use Demo Image"):
            use_demo = True


# Select final image to predict
final_image = None
final_image_name = None

if uploaded:
    final_image_name = uploaded.name
    final_image = Image.open(uploaded).convert("RGB")

elif use_demo:
    final_image_name = "demo_image.png"
    final_image = Image.open(DEMO_IMAGE_PATH).convert("RGB")


# Show preview
if final_image:
    st.image(final_image, caption="Selected Image", use_column_width=True)


# Predict button
if st.button("Predict"):
    if final_image is None:
        st.warning("Please upload an image or use demo image.")
    else:
        with st.spinner("Processing..."):
            preds = predict(model, device, final_image, inv_map, topk=3)

        st.success("Prediction Completed!")
        for name, prob in preds:
            st.write(f"**{name}** — {prob:.3f}")

        st.markdown("---")
        st.info("Prediction is based on a transfer-learning model trained on your crop pest dataset.")
