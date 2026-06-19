import os
import joblib
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

FEATURE_CSV = "Datasets/benchmark_features.csv"

MODEL_PATH = (
    "Models/forensic_gbt_filtered_tags.pkl"
)

OUTPUT_CSV = "arena_predictions.csv"

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading benchmark features...")

df = pd.read_csv(FEATURE_CSV)

print(f"Rows: {len(df)}")

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading model...")

loaded = joblib.load(MODEL_PATH)

if isinstance(loaded, dict) and "model" in loaded:

    model = loaded["model"]

    if "features" in loaded:
        features = loaded["features"]
    else:
        features = list(
            model.feature_names_in_
        )

else:

    model = loaded
    features = list(
        model.feature_names_in_
    )

print(f"Model uses {len(features)} features")

# ==========================================================
# VERIFY FEATURES
# ==========================================================

missing = [
    f
    for f in features
    if f not in df.columns
]

if missing:

    print()
    print("Missing features:")
    print(missing)

    raise ValueError(
        "Benchmark CSV missing required features."
    )

# ==========================================================
# PREDICT
# ==========================================================

X = df[features]

print("Running predictions...")

pred = model.predict(X)

if hasattr(model, "predict_proba"):

    prob = model.predict_proba(X)[:, 1]

else:

    print(
        "WARNING: Model has no predict_proba()"
    )

    prob = pred.astype(float)

# ==========================================================
# BUILD SUBMISSION FILE
# ==========================================================

submission = pd.DataFrame({
    "image_id": df["id"],
    "prediction": [
        "ai" if p == 1 else "real"
        for p in pred
    ],
    "confidence": prob
})

submission.to_csv(
    OUTPUT_CSV,
    index=False
)

# ==========================================================
# SUMMARY
# ==========================================================

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(
    f"Predictions saved to: {OUTPUT_CSV}"
)

print()
print(submission.head())