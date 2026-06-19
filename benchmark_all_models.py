import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

FEATURE_CSV = "Datasets/benchmark_features.csv"
MODELS_DIR = "Models"

print("Loading benchmark dataset...")
df = pd.read_csv(FEATURE_CSV)

y_true = df["label"]

results = []

print(f"Found {len(df)} benchmark images")
print()

for model_file in sorted(os.listdir(MODELS_DIR)):

    if not model_file.endswith(".pkl"):
        continue

    model_path = os.path.join(
        MODELS_DIR,
        model_file
    )

    print("=" * 80)
    print(model_file)
    print("=" * 80)

    try:

        loaded = joblib.load(model_path)

        # --------------------------------------------------
        # CASE 1
        # Saved as dict
        # --------------------------------------------------
        if (
            isinstance(loaded, dict)
            and "model" in loaded
        ):
            model = loaded["model"]

            if "features" in loaded:
                features = loaded["features"]
            else:
                features = list(
                    model.feature_names_in_
                )

        # --------------------------------------------------
        # CASE 2
        # Saved directly
        # --------------------------------------------------
        else:
            model = loaded
            features = list(
                model.feature_names_in_
            )

        missing = [
            f for f in features
            if f not in df.columns
        ]

        if missing:
            print(
                f"Missing {len(missing)} features"
            )
            print(missing)
            continue

        X = df[features]

        pred = model.predict(X)
        prob = model.predict_proba(X)[:, 1]

        acc = accuracy_score(
            y_true,
            pred
        )

        auc = roc_auc_score(
            y_true,
            prob
        )

        cm = confusion_matrix(
            y_true,
            pred
        )

        tn, fp, fn, tp = cm.ravel()

        results.append({
            "model": model_file,
            "accuracy": acc,
            "auc": auc,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "features": len(features)
        })

        print(f"Features : {len(features)}")
        print(f"Accuracy : {acc:.6f}")
        print(f"AUC      : {auc:.6f}")

    except Exception as e:

        print("FAILED")
        print(e)

print()
print("=" * 80)
print("FINAL LEADERBOARD")
print("=" * 80)

if len(results):

    leaderboard = pd.DataFrame(
        results
    ).sort_values(
        "accuracy",
        ascending=False
    )

    print(
        leaderboard.to_string(
            index=False
        )
    )

    leaderboard.to_csv(
        "benchmark_results.csv",
        index=False
    )

    print()
    print(
        "Saved benchmark_results.csv"
    )

else:

    print("No successful models.")