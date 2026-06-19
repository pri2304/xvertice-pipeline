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
import numpy as np

# 1. Prevent crashes on truncated/partial images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================= CONFIGURATION =================
CSV_FILE = "/home/pri/PycharmProjects/videotojpeg/updated_cnn_dataset_v3.csv"
MODEL_SAVE_PATH = "Models/efficientnet_b4_multihead_best.pth"

# EfficientNet-B4 Standards
IMG_SIZE = 380
BATCH_SIZE = 16
ACCUM_STEPS = 1
EPOCHS = 15
LEARNING_RATE = 0.0001
NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Loss Weight for Auxiliary Head
LAMBDA_NAT = 0.3

TAG_WEIGHTS = {
    "FlickrFace": 2.0,
    "FaceForensics_Real": 2.0,
    "FantasyID_Real": 2.0,
    "Smartphone": 4.0,
    "Social_media_laundered_real": 3.0,
    "Social_media_laundered_fake": 1.5
}


# =================================================

class MultiHeadEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load Backbone
        base = models.efficientnet_b4(weights='DEFAULT')

        # Extract features and pooling (everything before classifier)
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier_drop = base.classifier[0]  # Dropout

        in_features = base.classifier[1].in_features

        # Head 1: Real vs Fake (Binary Classification)
        self.head_real_fake = nn.Linear(in_features, 1)

        # Head 2: Camera Naturalness (Regression 0-1)
        self.head_camera_nat = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier_drop(x)

        # Logits for BCE Loss
        rf_logits = self.head_real_fake(x)

        # Score for MSE Loss (Sigmoid to bound 0-1)
        nat_score = torch.sigmoid(self.head_camera_nat(x))

        return rf_logits, nat_score


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

    def _get_naturalness_target(self, label, tag):
        tag = str(tag).lower()

        # Base Defaults
        if label == 0:  # Real
            score = 0.95

            # Heuristics for Real
            if any(x in tag for x in ['laundered', 'social', 'whatsapp', 'facebook', 'twitter', 'instagram']):
                score = 0.70
            elif any(x in tag for x in ['blur', 'night', 'low_light']):
                score = 0.75
            elif 'scan' in tag:
                score = 0.80
            else:
                score = 1.0  # Clean High Quality

        else:  # Fake
            score = 0.1

            # Heuristics for Fake
            if any(x in tag for x in ['edit', 'upscale', 'retouch', 'nano', 'grok']):
                score = 0.45  # AI Edited / Subtle
            elif any(x in tag for x in ['laundered', 'multipath']):
                # Deepfakes that are laundered might be slightly higher or lower?
                pass
            else:
                score = 0.0  # Pure GenAI

        return score

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self._resolve_path(row['path'])
        label = int(row['label'])
        tag = row.get('tag1', 'Unknown')

        if img_path is None: return None  # Filtered by collate_fn

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

            nat_target = self._get_naturalness_target(label, tag)

            return image, torch.tensor(label, dtype=torch.float32), torch.tensor(nat_target, dtype=torch.float32)
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
        return torch.Tensor(), torch.Tensor(), torch.Tensor()
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

    print(f"--- Forensics Training: MultiHead EfficientNet-B4 on {DEVICE} ---")

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

    # === EfficientNet Standard Normalization ===
    train_transforms = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.RandomCrop(IMG_SIZE),
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
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        ForensicDataset(val_df, val_transforms),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2
    )

    print("Initializing MultiHead EfficientNet-B4...")
    try:
        model = MultiHeadEfficientNet().to(DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # LOSS FUNCTIONS
    criterion_rf = nn.BCEWithLogitsLoss()
    criterion_nat = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # GradScaler for Mixed Precision
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_recall = 0.0

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", total=len(train_loader))

        for i, batch in enumerate(loop):
            # FIXED: Check if batch[0] (images) is empty (len==0)
            if batch is None or len(batch) < 3 or len(batch[0]) == 0:
                continue

            images, labels_rf, labels_nat = batch
            images = images.to(DEVICE)
            labels_rf = labels_rf.to(DEVICE).unsqueeze(1)
            labels_nat = labels_nat.to(DEVICE).unsqueeze(1)

            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    rf_logits, nat_score = model(images)

                    loss_rf = criterion_rf(rf_logits, labels_rf)
                    loss_nat = criterion_nat(nat_score, labels_nat)

                    total_loss = loss_rf + (LAMBDA_NAT * loss_nat)
                    total_loss = total_loss / ACCUM_STEPS

                scaler.scale(total_loss).backward()

                if (i + 1) % ACCUM_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Fallback for CPU
                rf_logits, nat_score = model(images)
                loss_rf = criterion_rf(rf_logits, labels_rf)
                loss_nat = criterion_nat(nat_score, labels_nat)
                total_loss = loss_rf + (LAMBDA_NAT * loss_nat)

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            loop.set_postfix(loss=total_loss.item() * ACCUM_STEPS,
                             rf=loss_rf.item(),
                             nat=loss_nat.item())

        # Validation
        model.eval()
        val_preds_rf, val_targets_rf = [], []

        with torch.no_grad():
            for batch in val_loader:
                # FIXED: Check if batch[0] (images) is empty
                if batch is None or len(batch) < 3 or len(batch[0]) == 0:
                    continue
                images, labels_rf, labels_nat = batch
                images = images.to(DEVICE)
                labels_rf = labels_rf.to(DEVICE).unsqueeze(1)

                rf_logits, nat_score = model(images)

                preds = torch.sigmoid(rf_logits) > 0.5
                val_preds_rf.extend(preds.cpu().numpy())
                val_targets_rf.extend(labels_rf.cpu().numpy())

        if len(val_targets_rf) > 0:
            rec = recall_score(val_targets_rf, val_preds_rf, zero_division=0)
            acc = accuracy_score(val_targets_rf, val_preds_rf)
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
        all_nat_scores = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # FIXED: Check if batch[0] (images) is empty
                if batch is None or len(batch) < 3 or len(batch[0]) == 0:
                    continue

                images, labels, tags = batch
                images = images.to(DEVICE)

                rf_logits, nat_score = model(images)

                preds = torch.sigmoid(rf_logits).cpu().numpy() > 0.5
                nat_vals = nat_score.cpu().numpy().flatten()

                all_preds.extend(preds.flatten().astype(int))
                all_labels.extend(labels.numpy().astype(int))
                all_tags.extend(tags)
                all_nat_scores.extend(nat_vals)

        if len(all_labels) > 0:
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))

            print("\nConfusion Matrix:")
            print(confusion_matrix(all_labels, all_preds))

            if len(all_tags) == len(all_labels):
                results = pd.DataFrame({
                    "Tag": all_tags,
                    "True": all_labels,
                    "Pred": all_preds,
                    "NatScore": all_nat_scores
                })
                results["Correct"] = results["True"] == results["Pred"]

                report = results.groupby("Tag").agg(
                    Count=('Correct', 'count'),
                    Accuracy=('Correct', 'mean'),
                    AvgNatScore=('NatScore', 'mean')
                ).sort_values(by="Accuracy")

                print("\n--- Tag-wise Performance & Naturalness ---")
                print(f"{'Tag':<35} | {'Count':<5} | {'Acc':<6} | {'NatScore':<8} | {'Status'}")
                print("-" * 80)
                for tag, row in report.iterrows():
                    acc = row['Accuracy'] * 100
                    nat = row['AvgNatScore']
                    status = "✅" if acc > 85 else "⚠️" if acc > 65 else "❌"
                    print(f"{tag:<35} | {row['Count']:<5} | {acc:.1f}%   | {nat:.3f}    | {status}")


if __name__ == "__main__":
    try:
        train_and_eval()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")