import os
import time
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFile

# --- 1. CONFIGURATION ---
INPUT_CSV = "/home/pri/PycharmProjects/benchmark-dataset/dataset_metadata/images_metadata.csv"
OUTPUT_CSV = "Datasets/benchmark_features.csv"
BENCHMARK_ROOT = "/home/pri/PycharmProjects/benchmark-dataset/"

# Model Config
CNN_MODEL_PATH = "Models/efficientnet_b4_multihead_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_WORKERS = 8
IMG_SIZE = 380

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 2. FORENSIC MODULE IMPORTS ---

from noise_analysis_test import NoiseAnalysis
from jpeg_ghost_analysis import GHOST
from metadata_analysis_final import Metadata
from dqt_aware_ela_test import ELA
from NLF import NLF
from DCT import DCT
from GAN_frequency import GANMonitor



# --- 3. MODEL DEFINITION ---
class MultiHeadEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b4(weights='DEFAULT')
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier_drop = base.classifier[0]
        in_features = base.classifier[1].in_features
        self.head_real_fake = nn.Linear(in_features, 1)
        self.head_camera_nat = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier_drop(x)
        rf_logits = self.head_real_fake(x)
        nat_score = torch.sigmoid(self.head_camera_nat(x))
        return rf_logits, nat_score


# Global Resources
cnn_model = None
cnn_trans = None


def init_resources():
    global cnn_model, cnn_trans
    print(f"Loading Model: {CNN_MODEL_PATH}")

    if not os.path.exists(CNN_MODEL_PATH):
        print(f"❌ CRITICAL: Model file NOT found at {CNN_MODEL_PATH}")
        return False

    try:
        model = MultiHeadEfficientNet()
        state_dict = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        cnn_model = model
        cnn_trans = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print(f"✅ Model Loaded on {DEVICE}")
        return True
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        return False


def get_cnn_inference(path):
    """Returns (cnn_score, nat_score). Prints error if fails."""
    if cnn_model is None: return None, None

    try:
        image = Image.open(path).convert("RGB")
        input_tensor = cnn_trans(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            rf_logits, nat_val = cnn_model(input_tensor)

        cnn_score = torch.sigmoid(rf_logits).item()
        nat_score = nat_val.item()

        return cnn_score, nat_score
    except Exception as e:
        # Print the specific error to the console so user sees WHY it failed
        tqdm.write(f"⚠️ Inference failed for {os.path.basename(path)}: {e}")
        return None, None


# --- 4. PROCESSING LOGIC ---

def process_update_row(row):
    """
    Updates existing row.
    IMPORTANT: Returns the ORIGINAL row if update fails, ensuring no data loss.
    """
    path = row.get('path')

    # 1. Check Path
    if not os.path.exists(path):
        tqdm.write(f"⚠️ File missing: {path}")
        return row  # Return original row unchanged

    # 2. Run Inference
    c_score, n_score = get_cnn_inference(path)

    if c_score is not None:
        # Success: Update values
        row['cnn_score'] = c_score
        row['nat_score'] = n_score
    else:
        # Failure: Keep old 'cnn_score' if exists, set nat_score to -1 (error flag)
        if 'nat_score' not in row:
            row['nat_score'] = -1.0

    return row


def safe_run_test(func, image_path):
    try:
        res = func(image_path)
        if isinstance(res, (tuple, list)): return res[0]
        return res
    except:
        return None


def flatten_result(prefix, data):
    flat = {}
    if data is None: return {}

    if not isinstance(data, dict):
        if isinstance(data, (int, float)):
            flat[f"{prefix}_val"] = float(data)
        return flat

    if prefix == "meta":
        flags = data.get("flags", {})
        info = data.get("data", {})
        flat["meta_stripped"] = 1.0 if flags.get("metadata_stripped") else 0.0
        flat["meta_software"] = 1.0 if flags.get("software_trace_found") else 0.0
        flat["meta_suspicious"] = 1.0 if flags.get("is_suspicious_hex") else 0.0
        flat["meta_thumb_mis"] = 1.0 if flags.get("thumbnail_mismatch") else 0.0
        flat["meta_deep_edit"] = 1.0 if flags.get("deep_edit_history") else 0.0
        flat["meta_res_mis"] = 1.0 if flags.get("resolution_mismatch") else 0.0
        flat["meta_time_mis"] = 1.0 if flags.get("timestamp_mismatch") else 0.0
        flat["meta_geo"] = 1.0 if flags.get("has_geo_tag") else 0.0
        flat["meta_icc"] = 1.0 if flags.get("has_icc_profile") else 0.0
        flat["meta_high_chroma"] = 1.0 if flags.get("high_chroma_sampling") else 0.0
        flat["meta_has_app0"] = 1.0 if flags.get("has_APP0") else 0.0
        flat["meta_has_app1"] = 1.0 if flags.get("has_APP1") else 0.0
        flat["meta_has_com"] = 1.0 if flags.get("has_COM") else 0.0
        flat["meta_valid_soi"] = 1.0 if flags.get("valid_soi") else 0.0
        flat["meta_valid_eoi"] = 1.0 if flags.get("valid_eoi") else 0.0
        flat["meta_hist_count"] = float(info.get("history_count", 0))
        return flat

    for key, val in data.items():
        if isinstance(val, (int, float, bool)):
            clean_key = key.replace("standard_deviation", "std").replace("mean_intensity", "mean")
            clean_key = clean_key.replace("average_variance", "avg_var").replace("high_freq_energy", "hf_energy")
            clean_key = clean_key.replace("power_law_fit_error", "fit_err")
            col_name = f"{prefix}_{clean_key}"
            flat[col_name] = float(val)
    return flat


def process_extract_new(row):
    """Runs full extraction for new rows."""
    path = row['filename']
    if not os.path.exists(path): return None

    start_time = time.time()

    features = {
        "id": row["id"],
        "image_path": path,
        "label": row["is_ai"],
        "generator": row["generator"],
        "category": row["category"]
    }

    # CNN
    c, n = get_cnn_inference(path)
    features['cnn_score'] = c
    features['nat_score'] = n

    # CPU Tests
    test_map = {
        "ela": ELA.ela, "ghost": GHOST.jpeg_ghost, "meta": Metadata.analyze,
        "exp": NoiseAnalysis.exposure_check, "lap": NoiseAnalysis.laplacian_check,
        "med": NoiseAnalysis.median_check, "loc": NoiseAnalysis.local_variance_check,
        "bag": NoiseAnalysis.block_boundary_check, "nlf": NLF.nlf_analyze,
        "dct": DCT.dct_analyze, "gan": GANMonitor.analyze
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=TEST_WORKERS) as executor:
        future_to_prefix = {
            executor.submit(safe_run_test, func, path): prefix
            for prefix, func in test_map.items()
        }

        for future in concurrent.futures.as_completed(future_to_prefix):
            try:
                res = future.result()
                features.update(flatten_result(future_to_prefix[future], res))
            except:
                pass

    features['time'] = round(time.time() - start_time, 4)
    return features


# --- 5. MAIN ---
def main():
    print("--- BENCHMARK FEATURE EXTRACTION ---")

    if not init_resources():
        return

    if not os.path.exists(INPUT_CSV):
        print(f"❌ Missing input CSV: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    df["filename"] = df["filename"].apply(
        lambda x: os.path.join(BENCHMARK_ROOT, x)
    )

    print(f"Loaded {len(df)} images")

    extracted_rows = []

    for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Extracting Features"
    ):
        try:
            res = process_extract_new(row)

            if res:
                extracted_rows.append(res)

                if len(extracted_rows) % 50 == 0:
                    pd.DataFrame(extracted_rows).to_csv(
                        OUTPUT_CSV,
                        index=False
                    )

        except Exception as e:
            tqdm.write(
                f"FAILED: {row['filename']}"
            )
            tqdm.write(str(e))

    pd.DataFrame(extracted_rows).to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Rows Saved: {len(extracted_rows)}")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()