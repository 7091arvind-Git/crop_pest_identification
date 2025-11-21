🌾 Crop Pest Identification (PyTorch)

Author: 7091arvind-Git
Hackathon: Samsung Innovation Campus – AI/ML
Framework: PyTorch + TorchVision
Model: MobileNetV2 (Transfer Learning)

📌 Overview

This project identifies crop pests from images using a lightweight MobileNetV2 CNN.
It is optimized for hackathon conditions:

Uses 20% subset for faster training

Transfer learning for higher accuracy

3 epochs only (fast execution)

Automatically saves model + class names

Includes auto-inference code that picks a test image

Works on CPU or CUDA

📁 Directory Structure

project/
│── train.py / notebook cells
│── saved_models/           # model.pth saved here
│── meta/                   # class_names.json stored here
│── dataset/                # class subfolders of images
│── README.md

Dataset structure:

dataset/
│── aphid/
│── beetle/
│── caterpillar/
│── healthy/

🚀 Features

MobileNetV2 pretrained on ImageNet

Custom classifier head

Frozen backbone for fast training

Image preprocessing: Resize → Flip → Normalize

Automatic test-image inference

Works with Windows path D:\sic\dataset

🔧 Setup Instructions
1. Clone the repo
git clone https://github.com/7091arvind-Git/crop_pest_identification.git
cd crop_pest_identification

2. Install dependencies

Create requirements.txt with:

torch
torchvision
Pillow
numpy


Install:

pip install -r requirements.txt

🧠 Training Workflow
📌 1. Dataset & Transforms

The training script loads your dataset:

DATA_DIR = r"D:\sic\dataset"


Transforms used:

transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

📌 2. Build 20% Dataset Subset

To make training fast:

n_subset = int(len(full_dataset) * 0.20)
subset = Subset(full_dataset, subset_indices)

📌 3. Model Setup (MobileNetV2)
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)


Backbone frozen:

for name, param in model.features.named_parameters():
    param.requires_grad = False

📌 4. Training (3 Epochs)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

📌 5. Saving Artifacts

Automatically saves:

saved_models/model.pth  
meta/class_names.json  

▶️ Run Training
python train.py


This will:

Load subset

Train MobileNetV2

Save model + metadata

Print loss per epoch

🖼️ Automatic Inference

A test image is selected automatically:

img_files = glob.glob(os.path.join(DATA_DIR, "**", "*.jpg"), recursive=True)


Model predicts class & probability:

Predicted: aphid (92.41%)

📤 Optional: Streamlit App

If you created app.py, start it with:

streamlit run app.py


Features of app:

Upload an image

Model predicts pest class

Shows confidence score

🎯 Why This Project is AI/ML

Real ML model trained

Transfer learning used

Proper dataset → preprocessing → model → training → inference pipeline

Saved model + metadata for deployment

📄 License

MIT License

📬 Contact

Author: 7091arvind-Git
GitHub: https://github.com/7091arvind-Git




