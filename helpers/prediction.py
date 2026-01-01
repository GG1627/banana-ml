import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
# import matplotlib.pyplot as plt
import os
import urllib.request
import sys

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
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(models_dir, exist_ok=True)  # Ensure models directory exists
        
        checkpoint_path = os.path.join(models_dir, "banana_ripness_predictor.pth")
        model_url = "https://github.com/GG1627/banana-ml/releases/download/v1.0.0/banana_ripness_predictor.pth"
        
        # Download model if it doesn't exist
        if not os.path.exists(checkpoint_path):
            print("📥 Model not found locally. Downloading from GitHub Releases...")
            print(f"   URL: {model_url}")
            
            try:
                # Download the model file with progress tracking
                def _download_progress_hook(count, block_size, total_size):
                    """Show download progress"""
                    percent = int(count * block_size * 100 / total_size)
                    sys.stdout.write(f"\r   Downloading: {percent}% ({count * block_size}/{total_size} bytes)")
                    sys.stdout.flush()
                
                urllib.request.urlretrieve(model_url, checkpoint_path, _download_progress_hook)
                print(f"\n✅ Model downloaded successfully to {checkpoint_path}")
                
                # Verify file was downloaded (check file size > 0)
                if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0:
                    raise FileNotFoundError("Downloaded file is empty or doesn't exist")
                    
            except urllib.error.HTTPError as e:
                error_msg = f"HTTP error {e.code}: Failed to download model from GitHub Releases"
                if e.code == 404:
                    error_msg += f"\n   Model file not found at: {model_url}"
                    error_msg += "\n   Please verify the release exists and the filename is correct."
                raise FileNotFoundError(error_msg) from e
            except urllib.error.URLError as e:
                raise FileNotFoundError(
                    f"Network error: Failed to download model from {model_url}\n"
                    f"   Error: {str(e)}\n"
                    f"   Please check your internet connection or the URL."
                ) from e
            except Exception as e:
                # Clean up partial download on error
                if os.path.exists(checkpoint_path):
                    try:
                        os.remove(checkpoint_path)
                    except:
                        pass
                raise FileNotFoundError(
                    f"Unexpected error downloading model: {str(e)}\n"
                    f"   URL: {model_url}"
                ) from e
        
        # Load the checkpoint
        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model file not found at: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=_device, weights_only=False)
            
            # Verify checkpoint has required keys
            if 'model_state_dict' not in checkpoint:
                raise ValueError("Checkpoint file is missing 'model_state_dict' key. File may be corrupted.")
            
            _model.load_state_dict(checkpoint['model_state_dict'])
            _model.eval()
            
            epoch = checkpoint.get('epoch', 'unknown')
            mae = checkpoint.get('mae', 'unknown')
            print(f"✅ Model loaded! (Epoch {epoch}, MAE: {mae})")
            
        except FileNotFoundError:
            raise  # Re-raise FileNotFoundError as-is
        except KeyError as e:
            raise ValueError(f"Checkpoint file is missing required key: {str(e)}. File may be corrupted.") from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model checkpoint from {checkpoint_path}\n"
                f"   Error: {str(e)}\n"
                f"   The file may be corrupted. Try deleting it to force re-download."
            ) from e
    
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

