import time
import shutil
import os
import concurrent.futures
import uuid
import pandas as pd
import torch
import joblib
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.background import BackgroundTasks
from torchvision import transforms
from PIL import Image

# --- 1. IMPORT MODULES ---
from noise_analysis_test import NoiseAnalysis
from jpeg_ghost_analysis import GHOST
from metadata_analysis_final import Metadata
from dqt_aware_ela_test import ELA
from NLF import NLF
from DCT import DCT
from GAN_frequency import GANMonitor

# --- 2. CONFIGURATION ---
DEVICE = "cpu"
torch.set_num_threads(1)

CNN_MODEL_PATH = "Models/efficientnet_b4_forensics_best.pth"
GBT_MODEL_PATH = "Models/forensic_gbt_model.pkl"
RESULTS_DIR = "results"

# Ensure results folder exists
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI()

# Mount the results directory so images are accessible via URL
# Access via: http://server_ip:8000/results/filename.png
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# Global Models
cnn_model = None
gbt_model = None
cnn_transforms = None
feature_columns = []


# --- 3. CLEANUP TASK ---
def remove_file(path: str):
    """Background task to remove files after response is sent."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Error cleaning up {path}: {e}")


# --- 4. HELPER: SAVE IMAGE TO DISK ---
def save_visualization(obj, base_filename, suffix):
    """
    Saves PIL Image or BytesIO to disk and returns the URL.
    """
    if obj is None:
        return None

    filename = f"{base_filename}_{suffix}.png"
    filepath = os.path.join(RESULTS_DIR, filename)

    try:
        # Case A: PIL Image
        if isinstance(obj, Image.Image):
            obj.save(filepath, format="PNG")
            return f"/results/{filename}"

        # Case B: BytesIO Buffer (Matplotlib)
        elif hasattr(obj, 'read'):
            obj.seek(0)
            with open(filepath, "wb") as f:
                f.write(obj.read())
            return f"/results/{filename}"

        return None
    except Exception as e:
        print(f"Error saving viz: {e}")
        return None


# --- 5. FLATTENER (Same as before) ---
def flatten_data(prefix, data):
    flat = {}
    if data is None or not isinstance(data, dict): return flat

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


# --- 6. STARTUP ---
@app.on_event("startup")
def load_resources():
    global cnn_model, gbt_model, cnn_transforms, feature_columns
    print("--- Server Startup ---")

    # Load CNN
    try:
        import torchvision.models as models
        import torch.nn as nn
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
        print("✅ CNN Loaded")
    except Exception as e:
        print(f"❌ CNN Error: {e}")

    # Load GBT
    try:
        gbt_model = joblib.load(GBT_MODEL_PATH)
        if hasattr(gbt_model, "feature_names_in_"):
            feature_columns = list(gbt_model.feature_names_in_)
        print("✅ GBT Loaded")
    except Exception as e:
        print(f"❌ GBT Error: {e}")


# --- 7. TEST RUNNER ---
def run_timed_test(test_name, test_func, image_path, request_id):
    """
    Runs test and saves visualization to disk.
    """
    start = time.time()
    try:
        result = test_func(image_path)

        data_payload = {}
        visual_url = None

        if isinstance(result, (tuple, list)):
            data_payload = result[0]
            if len(result) > 1:
                # Save Image to Disk and get URL
                visual_url = save_visualization(result[1], request_id, test_name.replace(" ", "_").lower())
        else:
            data_payload = result

        return {
            "test_name": test_name,
            "status": "success",
            "data": data_payload,
            "image_url": visual_url,  # URL path for frontend
            "time_taken": round(time.time() - start, 4)
        }
    except Exception as e:
        return {
            "test_name": test_name,
            "status": "error",
            "error": str(e),
            "time_taken": round(time.time() - start, 4)
        }


# --- 8. MAIN ENDPOINT ---
@app.post("/analyze")
def analyze_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Create unique ID for this request
    request_id = uuid.uuid4().hex
    unique_filename = f"temp_{request_id}_{file.filename}"

    try:
        with open(unique_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Add original file to cleanup list
        background_tasks.add_task(remove_file, unique_filename)

        server_start = time.time()
        full_report = {"filename": file.filename, "request_id": request_id}

        # A. CNN
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
                full_report["cnn_analysis"] = {"score": 0.0, "status": "error"}

        # B. CPU Tests
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Pass request_id to save images uniquely
            future_map = {executor.submit(run_timed_test, name, func, unique_filename, request_id): name for name, func
                          in job_list}
            for future in concurrent.futures.as_completed(future_map):
                res = future.result()
                raw_results[future_map[future]] = res

        full_report["forensic_tests"] = raw_results

        # C. GBT
        flat_features = {"cnn_score": cnn_score}
        for prefix, res_obj in raw_results.items():
            if res_obj["status"] == "success":
                flat_features.update(flatten_data(prefix, res_obj["data"]))

        input_df = pd.DataFrame([flat_features])
        if feature_columns:
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            input_df = input_df[feature_columns]

        if gbt_model:
            prob_fake = float(gbt_model.predict_proba(input_df)[:, 1][0])
            verdict = "FAKE" if prob_fake > 0.5 else "REAL"
            full_report["final_verdict"] = {
                "decision": verdict,
                "fake_probability": round(prob_fake * 100, 2),
                "confidence_score": round(prob_fake, 4)
            }

        full_report["total_time"] = round(time.time() - server_start, 4)
        return full_report

    except Exception as e:
        return {"status": "CRASH", "error": str(e)}

# To run: uvicorn server_new:app --host 0.0.0.0 --port 8000