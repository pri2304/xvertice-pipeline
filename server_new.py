import time
import shutil
import os
import concurrent.futures
import uuid
import pandas as pd
import torch
import joblib
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image

# --- 1. IMPORT FORENSIC MODULES ---
# Ensure these files are in the same directory
from noise_analysis_test import NoiseAnalysis
from jpeg_ghost_analysis import GHOST
from metadata_analysis_final import Metadata
from dqt_aware_ela_test import ELA
from NLF import NLF
from DCT import DCT
from GAN_frequency import GANMonitor

# --- 2. SERVER CONFIGURATION ---
# mimicking low-spec server
DEVICE = "cpu"
torch.set_num_threads(1)  # Prevent PyTorch from hogging all CPU cores

CNN_MODEL_PATH = "Models/efficientnet_b4_forensics_best.pth"
GBT_MODEL_PATH = "Models/forensic_gbt_model.pkl"

# Initialize App
app = FastAPI()

# Global Models
cnn_model = None
gbt_model = None
cnn_transforms = None
feature_columns = []  # To ensure correct GBT column order


# --- 3. HELPER: FLATTENER (Crucial for GBT) ---
def flatten_data(prefix, data):
    """Converts nested forensic results into flat GBT features."""
    flat = {}
    if data is None or not isinstance(data, dict):
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
            flat[f"{prefix}_{clean_key}"] = float(val)
    return flat


# --- 4. STARTUP EVENT ---
@app.on_event("startup")
def load_resources():
    global cnn_model, gbt_model, cnn_transforms, feature_columns
    print("--- Server Startup: Loading Models ---")

    # Load CNN
    try:
        # Re-using your extraction logic roughly
        import torchvision.models as models
        import torch.nn as nn

        # Define architecture manually if load_cnn_model isn't perfectly portable
        model = models.efficientnet_b4(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        cnn_model = model

        cnn_transforms = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print("✅ CNN Loaded (CPU Mode)")
    except Exception as e:
        print(f"❌ CNN Load Failed: {e}")

    # Load GBT
    try:
        gbt_model = joblib.load(GBT_MODEL_PATH)
        # Extract feature names if available to ensure order
        if hasattr(gbt_model, "feature_names_in_"):
            feature_columns = list(gbt_model.feature_names_in_)
        else:
            print("⚠️ GBT model has no feature names stored. Using fallback list.")
            # Fallback: List the features manually if needed, or rely on dict keys
        print("✅ GBT Loaded")
    except Exception as e:
        print(f"❌ GBT Load Failed: {e}")


# --- 5. TEST RUNNER ---
def run_timed_test(test_name, test_func, image_path):
    start = time.time()
    try:
        result = test_func(image_path)
        # Unwrap tuple returns
        data = result[0] if isinstance(result, (tuple, list)) else result
        return {
            "test_name": test_name,
            "status": "success",
            "data": data,
            "time_taken": round(time.time() - start, 4)
        }
    except Exception as e:
        return {
            "test_name": test_name,
            "status": "error",
            "error": str(e),
            "time_taken": round(time.time() - start, 4)
        }


# --- 6. MAIN ENDPOINT ---
@app.post("/analyze")
def analyze_image(file: UploadFile = File(...)):
    unique_filename = f"temp_{uuid.uuid4().hex}_{file.filename}"
    try:
        # Save File
        with open(unique_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        server_start = time.time()
        full_report = {"filename": file.filename}

        # A. Run CNN Prediction (Sequential - avoids threading issues with PyTorch)
        cnn_score = 0.0
        if cnn_model:
            try:
                img = Image.open(unique_filename).convert("RGB")
                img_t = cnn_transforms(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = cnn_model(img_t)
                    cnn_score = float(torch.sigmoid(out).item())
                full_report["cnn_analysis"] = {"score": cnn_score, "status": "success"}
            except Exception as e:
                full_report["cnn_analysis"] = {"score": 0.0, "status": "error", "msg": str(e)}

        # B. Run CPU Tests (Parallel)
        job_list = [
            ("ela", ELA.ela),
            ("ghost", GHOST.jpeg_ghost),
            ("meta", Metadata.analyze),
            ("exp", NoiseAnalysis.exposure_check),
            ("lap", NoiseAnalysis.laplacian_check),
            ("med", NoiseAnalysis.median_check),
            ("loc", NoiseAnalysis.local_variance_check),
            ("bag", NoiseAnalysis.block_boundary_check),
            ("nlf", NLF.nlf_analyze),
            ("dct", DCT.dct_analyze),
            ("gan", GANMonitor.analyze)
        ]

        raw_results = {}
        # Max workers = 2 to stay safe on a 2-core server
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_map = {executor.submit(run_timed_test, name, func, unique_filename): name for name, func in job_list}
            for future in concurrent.futures.as_completed(future_map):
                res = future.result()
                raw_results[future_map[future]] = res

        full_report["forensic_tests"] = raw_results

        # C. Prepare Data for GBT
        # 1. Flatten everything
        flat_features = {"cnn_score": cnn_score}
        for prefix, res_obj in raw_results.items():
            if res_obj["status"] == "success":
                flat_features.update(flatten_data(prefix, res_obj["data"]))

        # 2. Align with GBT Columns
        # Create DataFrame with 1 row
        input_df = pd.DataFrame([flat_features])

        # Ensure all required columns exist (fill missing with 0)
        if feature_columns:
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            # Sort columns to match training order EXACTLY
            input_df = input_df[feature_columns]

        # D. Run GBT Prediction
        if gbt_model:
            prob_fake = float(gbt_model.predict_proba(input_df)[:, 1][0])
            verdict = "FAKE" if prob_fake > 0.5 else "REAL"
            full_report["final_verdict"] = {
                "decision": verdict,
                "fake_probability": round(prob_fake * 100, 2),
                "confidence_score": round(prob_fake, 4)
            }
        else:
            full_report["final_verdict"] = {"error": "Model not loaded"}

        full_report["total_time"] = round(time.time() - server_start, 4)
        return full_report

    except Exception as e:
        return {"status": "CRASH", "error": str(e)}

    finally:
        if os.path.exists(unique_filename):
            os.remove(unique_filename)

# To run: uvicorn server_new:app --host 0.0.0.0 --port 8000