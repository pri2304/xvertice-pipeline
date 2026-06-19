import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# CONFIG

INPUT_CSV = "Datasets/gbt_training_features_flattened.csv"

TEST_SIZE = 0.15
VAL_SIZE = 0.15

REMOVE_TAGS = {
    "Spliced",
    "Blurry_impainting",
    "Low_Quality_Casia",
    "Low_Quality_Casia_Splice",
    "Low_Quality_spliced",
    "Low_quality_photos"
}

# LOAD DATA

print("\nLoading dataset...")

df = pd.read_csv(INPUT_CSV)

print(f"Original samples: {len(df):,}")

df = df[~df["tag1"].isin(REMOVE_TAGS)].copy()

print(f"Filtered samples: {len(df):,}")

ignore_cols = [
    "path",
    "label_str",
    "tag1",
    "time",
    "label"
]

feature_cols = [c for c in df.columns if c not in ignore_cols]

print(f"Feature count: {len(feature_cols)}")

X = df[feature_cols]
y = df["label"]
tags = df["tag1"]

# SPLIT ONCE

print("\nCreating train/val/test split...")

try:

    X_temp, X_test, y_temp, y_test, tags_temp, tags_test = train_test_split(
        X,
        y,
        tags,
        test_size=TEST_SIZE,
        stratify=tags,
        random_state=42
    )

    val_split_adjusted = VAL_SIZE / (1 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_split_adjusted,
        stratify=tags_temp,
        random_state=42
    )

except ValueError:

    print("Tag stratification failed, using label stratification.")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=42
    )

    val_split_adjusted = VAL_SIZE / (1 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_split_adjusted,
        stratify=y_temp,
        random_state=42
    )

print(f"Train: {len(X_train):,}")
print(f"Val:   {len(X_val):,}")
print(f"Test:  {len(X_test):,}")

# SINGLE FEATURE SWEEP

print("RUNNING SINGLE FEATURE SWEEP")

results = []

best_model = None
best_feature = None
best_auc = -1
best_acc = -1

for idx, feature in enumerate(feature_cols, start=1):

    print(
        f"\n[{idx}/{len(feature_cols)}] "
        f"Testing: {feature}"
    )

    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric=["logloss", "error"],
        early_stopping_rounds=50,
        n_jobs=-1
    )

    model.fit(
        X_train[[feature]],
        y_train,
        eval_set=[
            (X_train[[feature]], y_train),
            (X_val[[feature]], y_val)
        ],
        verbose=False
    )

    pred = model.predict(
        X_test[[feature]]
    )

    prob = model.predict_proba(
        X_test[[feature]]
    )[:, 1]

    acc = accuracy_score(
        y_test,
        pred
    )

    auc = roc_auc_score(
        y_test,
        prob
    )

    print(
        f"AUC={auc:.6f} | "
        f"ACC={acc:.6f}"
    )

    results.append({
        "Feature": feature,
        "Accuracy": acc,
        "AUC": auc
    })

    if auc > best_auc:

        best_auc = auc
        best_acc = acc
        best_feature = feature
        best_model = model

# RESULTS

results_df = pd.DataFrame(results)

print("\n")
print("=" * 80)
print("TOP 20 FEATURES BY AUC")
print("=" * 80)

print(
    results_df
    .sort_values("AUC", ascending=False)
    .head(20)
    .to_string(index=False)
)

print("\n")
print("=" * 80)
print("TOP 20 FEATURES BY ACCURACY")
print("=" * 80)

print(
    results_df
    .sort_values("Accuracy", ascending=False)
    .head(20)
    .to_string(index=False)
)

print("\n")
print("=" * 80)
print("BEST SINGLE FEATURE")
print("=" * 80)

print(f"Feature : {best_feature}")
print(f"AUC     : {best_auc:.6f}")
print(f"ACC     : {best_acc:.6f}")

# SAVE BEST MODEL

joblib.dump(
    {
        "model": best_model,
        "features": [best_feature],
        "auc": best_auc,
        "accuracy": best_acc
    },
    "Models/forensic_gbt_single_feature_best.pkl"
)

print("\nSaved:")
print("Models/forensic_gbt_single_feature_best.pkl")

results_df.to_csv(
    "single_feature_sweep_results.csv",
    index=False
)

print("Saved:")
print("single_feature_sweep_results.csv")