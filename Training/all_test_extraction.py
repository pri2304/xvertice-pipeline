import os
import time
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import torch

# --- 1. IMPORT YOUR TESTS ---
from noise_analysis_test import NoiseAnalysis
from jpeg_ghost_analysis import GHOST
from metadata_analysis_final import Metadata
from dqt_aware_ela_test import ELA
from NLF import NLF
from DCT import DCT
from GAN_frequency import GANMonitor

# --- 2. IMPORT CNN MODULE ---
try:
    from cnn_extraction import load_forensic_model, get_forensic_transforms, get_cnn_score
except ImportError:
    print("❌ ERROR: Could not import 'cnn_extraction.py'.")
    exit()

# ================= CONFIGURATION =================
INPUT_CSV = "/home/pri/PycharmProjects/videotojpeg/custom_forensic_dataset.csv"
OUTPUT_CSV = "gbt_training_features_flattened.csv"
CNN_MODEL_PATH = "../Models/efficientnet_b4_forensics_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_WORKERS = 8
# =================================================

cnn_model = None
cnn_trans = None


def init_resources():
    global cnn_model, cnn_trans
    try:
        cnn_model = load_forensic_model(CNN_MODEL_PATH, DEVICE)
        cnn_trans = get_forensic_transforms()
        print("✅ CNN Resources Loaded.")
    except Exception as e:
        print(f"❌ CNN Load Error: {e}")


def safe_run_test(func, image_path):
    """Runs a test. Returns None if it crashes."""
    try:
        res = func(image_path)
        if isinstance(res, (tuple, list)):
            return res[0]
        return res
    except Exception:
        return None


def flatten_result(prefix, data):
    """
    Converts dictionary output into flat CSV columns.
    Handles the robust Metadata flags structure.
    """
    flat = {}
    if data is None:
        return {}

    if not isinstance(data, dict):
        if isinstance(data, (int, float)):
            flat[f"{prefix}_val"] = float(data)
        return flat

    # --- UPDATED METADATA HANDLING ---
    if prefix == "meta":
        flags = data.get("flags", {})
        info = data.get("data", {})

        # 1. Critical Forensic Flags (1.0 = True, 0.0 = False)
        # We explicitly map the keys from your JSON to clean column names
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

        # 2. Structure Markers (APP0/APP1 only - Ignoring APP2-15)
        # APP1 is critical because Real photos usually have APP1 (Exif).
        # AI/Screenshots often miss it.
        flat["meta_has_app0"] = 1.0 if flags.get("has_APP0") else 0.0
        flat["meta_has_app1"] = 1.0 if flags.get("has_APP1") else 0.0
        flat["meta_has_com"] = 1.0 if flags.get("has_COM") else 0.0

        # 3. File Integrity (Start/End markers)
        flat["meta_valid_soi"] = 1.0 if flags.get("valid_soi") else 0.0
        flat["meta_valid_eoi"] = 1.0 if flags.get("valid_eoi") else 0.0

        # 4. Numeric Data
        flat["meta_hist_count"] = float(info.get("history_count", 0))

        return flat

    # --- GENERIC HANDLING (Noise, DCT, etc.) ---
    for key, val in data.items():
        if isinstance(val, (int, float, bool)):
            clean_key = key.replace("standard_deviation", "std").replace("mean_intensity", "mean")
            clean_key = clean_key.replace("average_variance", "avg_var").replace("high_freq_energy", "hf_energy")
            clean_key = clean_key.replace("power_law_fit_error", "fit_err")
            col_name = f"{prefix}_{clean_key}"
            flat[col_name] = float(val)
    return flat


def process_single_image(row_tuple):
    index, row = row_tuple
    path = row['path']
    filename = os.path.basename(path)

    if not os.path.exists(path):
        tqdm.write(f"⚠️ [MISSING] {filename}")
        return None

    try:
        start_time = time.time()

        # --- Base Info ---
        features = {
            "path": path,
            "label": row['label'],
            "label_str": row['label_str'],
            "tag1": row.get("tag1", "Unknown")
        }

        # --- CNN Score ---
        if cnn_model:
            try:
                features["cnn_score"] = float(get_cnn_score(path, cnn_model, cnn_trans, DEVICE))
            except:
                features["cnn_score"] = None
        else:
            features["cnn_score"] = None

        # --- CPU Tests ---
        test_map = {
            "ela": ELA.ela,
            "ghost": GHOST.jpeg_ghost,
            "meta": Metadata.analyze,
            "exp": NoiseAnalysis.exposure_check,
            "lap": NoiseAnalysis.laplacian_check,
            "med": NoiseAnalysis.median_check,
            "loc": NoiseAnalysis.local_variance_check,
            "bag": NoiseAnalysis.block_boundary_check,
            "nlf": NLF.nlf_analyze,
            "dct": DCT.dct_analyze,
            "gan": GANMonitor.analyze
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=TEST_WORKERS) as executor:
            future_to_prefix = {
                executor.submit(safe_run_test, func, path): prefix
                for prefix, func in test_map.items()
            }

            for future in concurrent.futures.as_completed(future_to_prefix):
                prefix = future_to_prefix[future]
                try:
                    raw_data = future.result()
                    flattened_cols = flatten_result(prefix, raw_data)
                    features.update(flattened_cols)
                except Exception:
                    pass

        features["time"] = round(time.time() - start_time, 4)
        return features

    except Exception as e:
        # --- ERROR PRINTING ---
        tqdm.write(f"❌ [CRASH] {filename} | Error: {e}")
        return None


def main():
    print(f"--- Forensic Extraction (Verbose Log) ---")

    if not os.path.exists(INPUT_CSV):
        print(f"File not found: {INPUT_CSV}")
        return

    init_resources()
    df = pd.read_csv(INPUT_CSV)
    print(f"Processing {len(df)} images...")

    all_results = []
    pbar = tqdm(df.iterrows(), total=len(df), unit="img")

    for row_tuple in pbar:
        result = process_single_image(row_tuple)

        if result:
            all_results.append(result)

            # --- LOGGING TO TERMINAL ---
            # Using tqdm.write prints a line above the progress bar (doesn't overwrite)
            filename = os.path.basename(result['path'])
            tqdm.write(f"✅ [{result['label_str']}] {filename} | {result['time']}s")

            if len(all_results) % 50 == 0:
                pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)

    if all_results:
        final_df = pd.DataFrame(all_results)

        # Sort columns
        first_cols = ['path', 'label', 'label_str', 'tag1', 'cnn_score']
        cols = [c for c in first_cols if c in final_df.columns]
        cols += [c for c in final_df.columns if c not in cols]

        final_df = final_df[cols]
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Saved {len(final_df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()