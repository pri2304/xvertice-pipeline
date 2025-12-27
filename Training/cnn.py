import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix
import os
import gc

# 1. Prevent crashes on truncated/partial images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================= CONFIGURATION =================
CSV_FILE = "../Datasets/final_cnn_dataset.csv"
MODEL_SAVE_PATH = "../Models/efficientnet_b4_forensics_best.pth"

# EfficientNet-B0 Standards
IMG_SIZE = 380  # Native size for B0
BATCH_SIZE = 16  # B0 is small, 32 fits easily
ACCUM_STEPS = 1
EPOCHS = 15
LEARNING_RATE = 0.0001
NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

TAG_WEIGHTS = {
    "FlickrFace": 2.0,
    "FaceForensics_Real": 2.0,
    "FantasyID_Real": 2.0,
    "Smartphone": 4.0,
    "Social_media_laundered_real": 3.0,
    "Social_media_laundered_fake": 1.5
}


# =================================================

class ForensicDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def _resolve_path(self, path):
        if os.path.exists(path): return path
        if pd.isna(path) or path is None: return None

        # Linux Case Sensitivity Check
        base, ext = os.path.splitext(path)
        candidates = [
            base + ext.lower(), base + ext.upper(),
            base + ".png", base + ".PNG",
            base + ".jpg", base + ".JPG",
            base + ".jpeg", base + ".JPEG",
            base + ".webp", base + ".WEBP"
        ]
        for c in candidates:
            if os.path.exists(c): return c
        return None

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self._resolve_path(row['path'])
        label = int(row['label'])

        if img_path is None: return None  # Filtered by collate_fn

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception:
            return None


class ForensicEvalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def _resolve_path(self, path):
        if os.path.exists(path): return path
        if pd.isna(path) or path is None: return None
        base, ext = os.path.splitext(path)
        candidates = [base + ext.lower(), base + ext.upper(), base + ".png", base + ".PNG", base + ".jpg",
                      base + ".JPG", base + ".jpeg", base + ".JPEG"]
        for c in candidates:
            if os.path.exists(c): return c
        return None

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = int(row['label'])
        tag = str(row.get('tag1', 'Unknown'))
        img_path = self._resolve_path(row['path'])

        if img_path is None: return None

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label, tag
        except:
            return None


# --- Custom Collate to skip None (Corrupt/Missing files) ---
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor() if len(batch) == 2 else (torch.Tensor(), torch.Tensor(), [])
    return torch.utils.data.dataloader.default_collate(batch)


def get_sampler(df):
    print("--- Configuring Weighted Sampler ---")
    weights = []
    for _, row in df.iterrows():
        tag = row.get('tag1', '')
        w = 1.0
        for key, val in TAG_WEIGHTS.items():
            if key in str(tag):
                w = val
                break
        weights.append(w)
    return WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))


def train_and_eval():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"--- Forensics Training: EfficientNet-B0 on {DEVICE} ---")

    if not os.path.exists(CSV_FILE):
        print(f"CRITICAL ERROR: {CSV_FILE} not found.")
        return

    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Clean Data
    df = df.dropna(subset=['path', 'label'])
    df['label'] = df['label'].astype(int)
    if 'tag1' not in df.columns: df['tag1'] = 'Unknown'

    try:
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['tag1'], random_state=42)
    except:
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # === CHANGED: EfficientNet Standard Normalization ===
    train_transforms = transforms.Compose([
        transforms.Resize((400, 400)),  # Resize slightly larger
        transforms.RandomCrop(IMG_SIZE),  # Crop to 224
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_sampler = get_sampler(train_df)

    train_loader = DataLoader(
        ForensicDataset(train_df, train_transforms),
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        ForensicDataset(val_df, val_transforms),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2
    )

    # === CHANGED: Model Loading for EfficientNet-B0 ===
    print("Initializing EfficientNet-B4...")
    try:
        # weights='DEFAULT' loads best available weights (IMAGENET1K_V1)
        model = models.efficientnet_b4(weights='DEFAULT')

        # Adjust Classifier Head
        # EfficientNet uses 'classifier' block, index 1 is the Linear layer
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 1)

        model = model.to(DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_recall = 0.0

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", total=len(train_loader))

        for i, batch in enumerate(loop):
            if batch is None or len(batch) == 0 or (isinstance(batch[0], torch.Tensor) and batch[0].shape[0] == 0):
                continue

            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)

            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss = loss / ACCUM_STEPS

                scaler.scale(loss).backward()

                if (i + 1) % ACCUM_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Fallback for CPU
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            loop.set_postfix(loss=loss.item() * ACCUM_STEPS)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None or len(batch) == 0 or (isinstance(batch[0], torch.Tensor) and batch[0].shape[0] == 0):
                    continue
                images, labels = batch
                images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)

                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        if len(val_targets) > 0:
            rec = recall_score(val_targets, val_preds, zero_division=0)
            acc = accuracy_score(val_targets, val_preds)
            print(f"Val Acc: {acc:.4f} | Recall: {rec:.4f}")

            if rec >= best_recall:
                best_recall = rec
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print("  -> Saved Best Model")

    # FINAL EVALUATION
    if os.path.exists(MODEL_SAVE_PATH):
        print("\n" + "=" * 60)
        print("FINAL EVALUATION (BEST MODEL)")
        print("=" * 60)

        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model.eval()

        eval_loader = DataLoader(
            ForensicEvalDataset(val_df, val_transforms),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=min(NUM_WORKERS, os.cpu_count() or 1),
            collate_fn=collate_fn
        )

        all_preds, all_labels, all_tags = [], [], []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                if batch is None or len(batch) < 3: continue

                images, labels, tags = batch
                images = images.to(DEVICE)
                outputs = model(images)
                preds = torch.sigmoid(outputs).cpu().numpy() > 0.5

                all_preds.extend(preds.flatten().astype(int))
                all_labels.extend(labels.numpy().astype(int))
                all_tags.extend(tags)

        if len(all_labels) > 0:
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))

            print("\nConfusion Matrix:")
            print(confusion_matrix(all_labels, all_preds))

            if len(all_tags) == len(all_labels):
                results = pd.DataFrame({"Tag": all_tags, "True": all_labels, "Pred": all_preds})
                results["Correct"] = results["True"] == results["Pred"]
                report = results.groupby("Tag").agg(Count=('Correct', 'count'),
                                                    Accuracy=('Correct', 'mean')).sort_values(by="Accuracy")

                print("\n--- Tag-wise Performance ---")
                for tag, row in report.iterrows():
                    acc = row['Accuracy'] * 100
                    status = "✅" if acc > 85 else "⚠️" if acc > 65 else "❌"
                    print(f"{tag:<35} | {row['Count']:<5} | {acc:.1f}%    | {status}")


if __name__ == "__main__":
    try:
        train_and_eval()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")