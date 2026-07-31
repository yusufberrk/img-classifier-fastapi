import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import io

# Model Mimarisi (Eğitimdeki mimari ile birebir aynı olmalı)
class ObjectDetector(nn.Module):
    def __init__(self, num_classes):
        super(ObjectDetector, self).__init__()
        base_model = resnet50(weights=None)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.regressor = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        class_logits = self.classifier(x)
        bounding_box = self.regressor(x)
        return class_logits, bounding_box

# Cihazı CPU olarak ayarla (Docker içinde standart çıkarım için)
device = torch.device("cpu")
class_names = ['apple', 'banana', 'orange']

# Modeli başlat ve ağırlıkları yükle
model = ObjectDetector(num_classes=3)
# map_location=device ekliyoruz çünkü model GPU'da eğitilmiş olabilir, Docker'da CPU kullanacağız
model.load_state_dict(torch.load("app/object_detector_model.pth", map_location=device))
model.to(device)
model.eval()

# Çıkarım için sadece gerekli olan dönüşümler
transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_bytes):
    """Gelen byteları resme çevirir ve JSON dönecek şekilde tahmin yapar"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_logits, _ = model(input_tensor)
        probabilities = F.softmax(pred_logits, dim=1)
        max_prob, predicted_class_idx = torch.max(probabilities, 1)
        
        prob_value = max_prob.item()
        label = class_names[predicted_class_idx.item()]
        
    return {"label": label, "probability": float(prob_value)}