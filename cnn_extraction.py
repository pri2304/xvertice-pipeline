import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


# ================= HELPER FUNCTIONS =================

def load_forensic_model(model_path, device):
    """
    Loads the EfficientNet-B4 architecture and weights.
    Call this ONCE at the start of your script.
    """
    print(f"Loading CNN from {model_path}...")

    # 1. Setup Architecture (EfficientNet-B4)
    model = models.efficientnet_b4(weights=None)

    # 2. Match the Head (Linear layer for binary class)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)

    # 3. Load Weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"❌ Error: Model file '{model_path}' not found.")
        return None
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return None

    model = model.to(device)
    model.eval()  # Critical: Disables Dropout/BatchNorm training behavior
    return model


def get_forensic_transforms(img_size=380):
    """
    Returns the exact validation transforms used during training.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_cnn_score(image_path, model, transform, device):
    """
    Runs inference on a single image and returns the Fake Probability (0.0 to 1.0).
    """
    if not os.path.exists(image_path):
        return None

    try:
        # Open and Convert
        image = Image.open(image_path).convert("RGB")

        # Transform and Add Batch Dimension (3, H, W) -> (1, 3, H, W)
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(img_tensor)
            # Sigmoid converts logits -> probability (0.0 - 1.0)
            probability = torch.sigmoid(output).item()

        return probability

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


# ================= TESTER (RUN DIRECTLY) =================
if __name__ == "__main__":
    # Settings
    MODEL_FILE = "Models/efficientnet_b4_forensics_best.pth"
    TEST_IMAGE = "Testing Images/testcase12.jpeg"  # Change this to a real file to test
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Init
    cnn_model = load_forensic_model(MODEL_FILE, DEVICE)
    cnn_transforms = get_forensic_transforms(img_size=380)

    # 2. Run
    if cnn_model:
        score = get_cnn_score(TEST_IMAGE, cnn_model, cnn_transforms, DEVICE)

        if score is not None:
            print(f"\nImage: {TEST_IMAGE}")
            print(f"CNN Score: {score:.4f} (0=Real, 1=Fake)")
            print(f"Verdict:   {'FAKE' if score > 0.5 else 'REAL'}")