import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template_string
import uuid

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = 128
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------
# TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ----------------------------
# MODEL DEFINITION
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

# ----------------------------
# CLASS NAMES
# ----------------------------
class_names = ['animal', 'cartoon', 'floral', 'geometry', 'ikat', 'plain', 'polka dot', 'squares', 'stripes', 'tribal']
model = PatternCNN(num_classes=len(class_names)).to(DEVICE)
model.load_state_dict(torch.load("fabric_model.pth", map_location=DEVICE))
model.eval()

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
    return class_names[pred.item()]

# ----------------------------
# FLASK APP
# ----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PatternSense - Fabric Pattern Classifier</title>
  <style>
    body {
      margin: 0;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(to right, #f8f9fa, #e9ecef);
      color: #343a40;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
    }

    .container {
      background: #ffffff;
      padding: 40px;
      border-radius: 12px;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
      max-width: 500px;
      width: 90%;
      text-align: center;
    }

    h2 {
      margin-bottom: 30px;
      color: #007bff;
    }

    input[type="file"] {
      margin-bottom: 20px;
    }

    input[type="submit"] {
      background-color: #007bff;
      border: none;
      color: white;
      padding: 12px 24px;
      text-decoration: none;
      margin: 10px 0;
      cursor: pointer;
      border-radius: 6px;
      font-size: 16px;
    }

    .result {
      margin-top: 30px;
      font-size: 18px;
      color: #28a745;
      font-weight: bold;
    }

    img {
      margin-top: 20px;
      border-radius: 8px;
      max-width: 100%;
      height: auto;
      border: 1px solid #dee2e6;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>🧵 PatternSense Fabric Classifier</h2>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file" required><br>
      <input type="submit" value="Classify Pattern">
    </form>

    {% if result %}
      <div class="result">Prediction: {{ result }}</div>
      <img src="{{ image_url }}" alt="Uploaded Image">
    {% endif %}
  </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    image_url = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = predict_image(filepath)
            image_url = f"/{UPLOAD_FOLDER}/{filename}"
    return render_template_string(HTML_TEMPLATE, result=result, image_url=image_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return app.send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ----------------------------
# RUN APP
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)
