import time
import shutil
import os
import concurrent.futures
import uuid
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import joblib
import shap
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

CNN_MODEL_PATH = "Models/efficientnet_b4_multihead_best.pth"
GBT_MODEL_PATH = "Models/xgb_pathB_controlled_exposure_fixed.pkl"
RESULTS_DIR = "Results"
IMG_SIZE = 380

THRESH_FAKE = 0.95
THRESH_REAL = 0.30
MIN_FAKE_PROB_FOR_REAL = 0.15
SHAP_IMPACT_EPS = 0.01

os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI()
app.mount("/Results", StaticFiles(directory=RESULTS_DIR), name="Results")

# Global Models
cnn_model = None
gbt_model = None
gbt_explainer = None
cnn_transforms = None
feature_columns = []

# --- SHAP FEATURE GROUPS ---
SHAP_FEATURE_GROUPS = {
    "cnn": {
        "cnn_score",
        "nat_score"
    },

    "metadata": {
        "meta_stripped",
        "meta_software",
        "meta_suspicious",
        "meta_thumb_mis",
        "meta_deep_edit",
        "meta_res_mis",
        "meta_time_mis",
        "meta_geo",
        "meta_icc",
        "meta_high_chroma",
        "meta_has_app0",
        "meta_has_app1",
        "meta_has_com",
        "meta_valid_soi",
        "meta_valid_eoi",
        "meta_hist_count"
    },

    "noise_sensory": {
        "nlf",
        "lap",
        "med",
        "loc",
        "bag"
    },

    "compression_frequency": {
        "dct",
        "ela",
        "gan",
        "ghost"
    }
}

# --- 3. MODEL DEFINITION ---
class MultiHeadEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b4(weights=None)
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


# --- 4. HELPER FUNCTIONS ---
def remove_file(path: str):
    try:
        if os.path.exists(path): os.remove(path)
    except Exception as e:
        print(f"Error cleaning up {path}: {e}")


def save_visualization(obj, base_filename, suffix):
    if obj is None: return None
    filename = f"{base_filename}_{suffix}.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    try:
        if isinstance(obj, Image.Image):
            obj.save(filepath, format="PNG")
            return f"/results/{filename}"
        elif hasattr(obj, 'read'):
            obj.seek(0)
            with open(filepath, "wb") as f:
                f.write(obj.read())
            return f"/results/{filename}"
        return None
    except Exception as e:
        print(f"Error saving viz: {e}")
        return None

def compute_shap_summary(shap_map):
    """
    Step 6: Aggregate SHAP behavior for audit / diagnostics.
    Does NOT affect verdict.
    """
    impacts = [abs(v) for v in shap_map.values()]

    if not impacts:
        return {
            "mean_abs_impact": 0.0,
            "max_abs_impact": 0.0,
            "active_feature_count": 0
        }

    return {
        "mean_abs_impact": round(sum(impacts) / len(impacts), 6),
        "max_abs_impact": round(max(impacts), 6),
        "active_feature_count": sum(1 for v in impacts if v >= SHAP_IMPACT_EPS)
    }

def compute_grouped_shap_summary(shap_map):
    """
    Aggregates SHAP impacts by logical feature groups.
    Does NOT affect verdict.
    """
    grouped = {}

    for group_name, feature_keys in SHAP_FEATURE_GROUPS.items():
        impacts = []

        for feat, val in shap_map.items():
            for key in feature_keys:
                if feat == key or feat.startswith(f"{key}_"):
                    impacts.append(abs(val))
                    break

        if impacts:
            grouped[group_name] = {
                "mean_abs_impact": round(sum(impacts) / len(impacts), 6),
                "max_abs_impact": round(max(impacts), 6),
                "active_feature_count": sum(1 for v in impacts if v >= SHAP_IMPACT_EPS)
            }
        else:
            grouped[group_name] = {
                "mean_abs_impact": 0.0,
                "max_abs_impact": 0.0,
                "active_feature_count": 0
            }

    return grouped

def compute_directional_shap_metrics(shap_map):
    """
    Computes directional SHAP energy metrics.
    Does NOT affect verdict.
    """
    pos_sum = 0.0
    neg_sum = 0.0

    for v in shap_map.values():
        if v > 0:
            pos_sum += v
        elif v < 0:
            neg_sum += -v

    total = pos_sum + neg_sum
    balance = abs(pos_sum - neg_sum) / total if total > 0 else 0.0

    return {
        "pos_sum": round(pos_sum, 6),
        "neg_sum": round(neg_sum, 6),
        "total": round(total, 6),
        "balance": round(balance, 6)
    }

def compute_grouped_directional_shap_metrics(shap_map):
    """
    Computes directional SHAP metrics per feature group.
    """
    grouped_metrics = {}

    for group_name, feature_keys in SHAP_FEATURE_GROUPS.items():
        pos_sum = 0.0
        neg_sum = 0.0

        for feat, val in shap_map.items():
            for key in feature_keys:
                if feat == key or feat.startswith(f"{key}_"):
                    if val > 0:
                        pos_sum += val
                    elif val < 0:
                        neg_sum += -val
                    break

        total = pos_sum + neg_sum
        balance = abs(pos_sum - neg_sum) / total if total > 0 else 0.0

        grouped_metrics[group_name] = {
            "pos_sum": round(pos_sum, 6),
            "neg_sum": round(neg_sum, 6),
            "total": round(total, 6),
            "balance": round(balance, 6)
        }

    return grouped_metrics

# --- 5. FLATTENER (Used for Prediction Input) ---
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
            clean_key = key.replace("standard_deviation", "std").replace("mean_intensity", "mean") \
                .replace("average_variance", "avg_var").replace("high_freq_energy", "hf_energy") \
                .replace("power_law_fit_error", "fit_err")
            flat[f"{prefix}_{clean_key}"] = float(val)
    return flat


# --- 6. SHAP INJECTOR (NEW) ---
def inject_shap_impacts(full_report, shap_map):
    """
    Traverses the full_report structure and replaces raw values with
    { value: x, impact: y } objects using the shap_map.
    """

    # 1. CNN Analysis
    if "cnn_analysis" in full_report and full_report["cnn_analysis"].get("status") == "success":
        data = full_report["cnn_analysis"]
        if "score" in data:
            data["score"] = {"value": data["score"], "impact": shap_map.get("cnn_score", 0.0)}
        if "naturalness" in data:
            data["naturalness"] = {"value": data["naturalness"], "impact": shap_map.get("nat_score", 0.0)}

    # 2. Forensic Tests
    tests = full_report.get("forensic_tests", {})

    for prefix, test_res in tests.items():
        if test_res.get("status") != "success": continue

        data = test_res.get("data", {})

        # Special Handling for Metadata (Nested structure)
        if prefix == "meta":
            # Map Flags
            flags = data.get("flags", {})
            flag_mapping = {
                "metadata_stripped": "meta_stripped",
                "software_trace_found": "meta_software",
                "is_suspicious_hex": "meta_suspicious",
                "thumbnail_mismatch": "meta_thumb_mis",
                "deep_edit_history": "meta_deep_edit",
                "resolution_mismatch": "meta_res_mis",
                "timestamp_mismatch": "meta_time_mis",
                "has_geo_tag": "meta_geo",
                "has_icc_profile": "meta_icc",
                "high_chroma_sampling": "meta_high_chroma",
                "has_APP0": "meta_has_app0",
                "has_APP1": "meta_has_app1",
                "has_COM": "meta_has_com",
                "valid_soi": "meta_valid_soi",
                "valid_eoi": "meta_valid_eoi"
            }
            for orig_key, flat_key in flag_mapping.items():
                if orig_key in flags:
                    flags[orig_key] = {
                        "value": flags[orig_key],
                        "impact": shap_map.get(flat_key, 0.0)
                    }

            # Map Info (History Count)
            info = data.get("data", {})  # Yes, data.data exists in meta structure
            if "history_count" in info:
                info["history_count"] = {
                    "value": info["history_count"],
                    "impact": shap_map.get("meta_hist_count", 0.0)
                }

        else:
            # Generic Flat Tests (ELA, GAN, etc.)
            # We must replicate the key transformation logic to find the match
            new_data = {}
            for key, val in data.items():
                if isinstance(val, (int, float, bool)):
                    # Reconstruct the flat key used in training
                    clean_key = key.replace("standard_deviation", "std").replace("mean_intensity", "mean") \
                        .replace("average_variance", "avg_var").replace("high_freq_energy", "hf_energy") \
                        .replace("power_law_fit_error", "fit_err")
                    flat_key = f"{prefix}_{clean_key}"

                    new_data[key] = {
                        "value": val,
                        "impact": shap_map.get(flat_key, 0.0)
                    }
                else:
                    new_data[key] = val
            test_res["data"] = new_data

    return full_report


# --- 7. STARTUP ---
@app.on_event("startup")
def load_resources():
    global cnn_model, gbt_model, gbt_explainer, cnn_transforms, feature_columns
    print("--- Server Startup ---")

    try:
        model = MultiHeadEfficientNet()
        state_dict = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        cnn_model = model
        cnn_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print("✅ CNN Loaded")
    except Exception as e:
        print(f"❌ CNN Error: {e}")

    try:
        gbt_model = joblib.load(GBT_MODEL_PATH)
        if hasattr(gbt_model, "feature_names_in_"):
            feature_columns = list(gbt_model.feature_names_in_)
        gbt_explainer = shap.TreeExplainer(gbt_model)
        print("✅ GBT & SHAP Loaded")
    except Exception as e:
        print(f"❌ GBT Error: {e}")


# --- 8. RUNNER ---
def run_timed_test(test_name, test_func, image_path, request_id):
    start = time.time()
    try:
        result = test_func(image_path)
        data_payload = {}
        visual_url = None
        if isinstance(result, (tuple, list)):
            data_payload = result[0]
            if len(result) > 1:
                visual_url = save_visualization(result[1], request_id, test_name.replace(" ", "_").lower())
        else:
            data_payload = result
        return {"test_name": test_name, "status": "success", "data": data_payload, "image_url": visual_url,
                "time_taken": round(time.time() - start, 4)}
    except Exception as e:
        return {"test_name": test_name, "status": "error", "error": str(e), "time_taken": round(time.time() - start, 4)}


# --- 9. ENDPOINT ---
@app.post("/analyze")
def analyze_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    request_id = uuid.uuid4().hex
    unique_filename = f"temp_{request_id}_{file.filename}"

    try:
        with open(unique_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        background_tasks.add_task(remove_file, unique_filename)
        server_start = time.time()
        full_report = {"filename": file.filename, "request_id": request_id}

        # CNN
        cnn_score, nat_score = 0.0, 0.0
        if cnn_model:
            try:
                img = Image.open(unique_filename).convert("RGB")
                img_t = cnn_transforms(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    r, n = cnn_model(img_t)
                    cnn_score, nat_score = float(torch.sigmoid(r).item()), float(n.item())
                full_report["cnn_analysis"] = {"score": cnn_score, "naturalness": nat_score, "status": "success"}
            except Exception as e:
                full_report["cnn_analysis"] = {"status": "error", "error": str(e)}

        # CPU Tests
        job_list = [
            ("ela", ELA.ela), ("ghost", GHOST.jpeg_ghost), ("meta", Metadata.analyze),
            ("exp", NoiseAnalysis.exposure_check), ("lap", NoiseAnalysis.laplacian_check),
            ("med", NoiseAnalysis.median_check), ("loc", NoiseAnalysis.local_variance_check),
            ("bag", NoiseAnalysis.block_boundary_check), ("nlf", NLF.nlf_analyze),
            ("dct", DCT.dct_analyze), ("gan", GANMonitor.analyze)
        ]
        raw_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_map = {executor.submit(run_timed_test, n, f, unique_filename, request_id): n for n, f in job_list}
            for future in concurrent.futures.as_completed(future_map):
                raw_results[future_map[future]] = future.result()
        full_report["forensic_tests"] = raw_results

        # GBT & SHAP Injection
        flat_features = {"cnn_score": cnn_score, "nat_score": nat_score}
        for p, r in raw_results.items():
            if r["status"] == "success": flat_features.update(flatten_data(p, r["data"]))

        input_df = pd.DataFrame([flat_features])
        if feature_columns:
            for c in feature_columns:
                if c not in input_df.columns: input_df[c] = 0.0
            input_df = input_df[feature_columns]

        if gbt_model:
            prob_fake = float(gbt_model.predict_proba(input_df)[:, 1][0])

            if prob_fake >= THRESH_FAKE:
                decision = "FAKE"

            elif prob_fake <= THRESH_REAL:
                if prob_fake < MIN_FAKE_PROB_FOR_REAL:
                    decision = "LIKELY_REAL"
                else:
                    decision = "SUSPICIOUS"

            else:
                decision = "SUSPICIOUS"

            full_report["final_verdict"] = {
                "decision": decision,
                "fake_probability": round(prob_fake * 100, 2),
            }

            # --- SHAP CALCULATION ---
            if gbt_explainer:
                try:
                    shap_vals = gbt_explainer.shap_values(input_df)
                    vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

                    # Create Map: {feature_name: impact}
                    shap_map = {name: round(float(val), 4) for name, val in zip(feature_columns, vals)}

                    # Inject into Tree
                    full_report = inject_shap_impacts(full_report, shap_map)

                    # --- STEP 6: SHAP AGGREGATE SIGNALS ---
                    full_report["shap_summary"] = compute_shap_summary(shap_map)

                    # --- SHAP GROUPED SUMMARY ---
                    full_report["shap_group_summary"] = compute_grouped_shap_summary(shap_map)

                    # --- SHAP DIRECTIONAL METRICS ---
                    full_report["shap_directional_summary"] = {
                        "global": compute_directional_shap_metrics(shap_map),
                        "by_group": compute_grouped_directional_shap_metrics(shap_map)
                    }

                except Exception as e:
                    print(f"SHAP Error: {e}")

        full_report["total_time"] = round(time.time() - server_start, 4)
        return full_report

    except Exception as e:
        return {"status": "CRASH", "error": str(e)}




