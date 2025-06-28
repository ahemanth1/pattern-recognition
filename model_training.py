import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from PIL import Image

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_DIR =  "/Users/vejandlaanji/Documents/smartintern/data_pattern" # ✅ Change this
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # M1/M2/M3 uses Metal (MPS)

# ----------------------------
# DATA TRANSFORMS
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = train_dataset.classes

# Train/Val Split
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ----------------------------
# MODEL DEFINITION (Custom CNN)
# ----------------------------
class PatternCNN(nn.Module):
    def __init__(self, num_classes):
        super(PatternCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = PatternCNN(num_classes=len(class_names)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# TRAINING LOOP
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

# ----------------------------
# EVALUATION
# ----------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_image(img_path):
    model.eval()
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
    print(f"Predicted: {class_names[pred.item()]}")
    return class_names[pred.item()]

# ----------------------------
# Test Prediction
# ----------------------------
# test_image = "/path/to/test.jpg"
# predict_image(test_image)
torch.save(model.state_dict(), "fabric_model.pth")
