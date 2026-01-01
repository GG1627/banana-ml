import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
# import matplotlib.pyplot as plt
import os

# Model definition (needed to load the state dict)
class BananaViTRegressor(nn.Module):
    def __init__(self):
        super(BananaViTRegressor, self).__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.regressor = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        features = self.vit(x)
        return self.regressor(features)

# Global model variable (lazy loaded)
_model = None
_device = None

def _load_model():
    """Lazy load the model on first use"""
    global _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = BananaViTRegressor().to(_device)
        
        # Get path relative to this file's location (works regardless of working directory)
        checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "banana_ripness_predictor.pth")
        checkpoint = torch.load(checkpoint_path, map_location=_device, weights_only=False)
        _model.load_state_dict(checkpoint['model_state_dict'])
        _model.eval()
        
        print(f"✅ Model loaded! (Epoch {checkpoint['epoch']}, MAE: {checkpoint['mae']:.4f})")
    
    return _model, _device

# Prediction function
def predict(image_path):
    """Predict days until banana is rotten"""
    model, device = _load_model()
    
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(image_tensor).item()
    
    return max(0, prediction)

# Example usage
if __name__ == "__main__":
    # Test on example image
    image_path = "example_images/img0.jpg"
    
    if os.path.exists(image_path):
        days_left = predict(image_path)
        
        # Show image with prediction
        image = Image.open(image_path)
        # plt.imshow(image)
        # plt.title(f"Predicted days left until rotten: {days_left:.2f} days")
        # plt.axis('off')
        # plt.show()
        
        print(f"\n🍌 Prediction: {days_left:.2f} days until rotten")
        
        # Status interpretation
        if days_left > 7:
            print("Status: Very fresh! 🟢")
        elif days_left > 3:
            print("Status: Fresh 🟡")
        elif days_left > 0:
            print("Status: Getting ripe 🟠")
        else:
            print("Status: Might be rotten already 🔴")
    else:
        print(f"Image not found: {image_path}")

