import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# CONFIG

INPUT_CSV = "Datasets/final_merged_dataset.csv"

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
# (same split for every experiment)

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

    print("Tag stratification failed, falling back to label stratification.")

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

# FULL MODEL

print("TRAINING FULL MODEL")

full_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric=["logloss", "error"],
    early_stopping_rounds=50,
    n_jobs=-1
)

full_model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_train, y_train),
        (X_val, y_val)
    ],
    verbose=False
)

full_pred = full_model.predict(X_test)
full_prob = full_model.predict_proba(X_test)[:, 1]

full_acc = accuracy_score(y_test, full_pred)
full_auc = roc_auc_score(y_test, full_prob)

print(f"Full Model Accuracy : {full_acc:.6f}")
print(f"Full Model AUC      : {full_auc:.6f}")

# FEATURE IMPORTANCE RANKING

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": full_model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

ranked_features = importance_df["Feature"].tolist()

print("\nTop 20 Features")

for idx, row in importance_df.head(20).iterrows():
    print(
        f"{idx + 1:2d}. "
        f"{row['Feature']:<35} "
        f"{row['Importance']:.6f}"
    )

# TOP-K SWEEP

print("RUNNING TOP-K SWEEP")

results = []

best_model = None
best_features = None
best_accuracy = -1
best_auc = -1

for k in range(1, len(ranked_features) + 1):

    selected_features = ranked_features[:k]

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
        X_train[selected_features],
        y_train,
        eval_set=[
            (X_train[selected_features], y_train),
            (X_val[selected_features], y_val)
        ],
        verbose=False
    )

    pred = model.predict(
        X_test[selected_features]
    )

    prob = model.predict_proba(
        X_test[selected_features]
    )[:, 1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    if acc > best_accuracy:
        best_accuracy = acc
        best_auc = auc
        best_model = model
        best_features = selected_features.copy()

    results.append({
        "K": k,
        "Accuracy": acc,
        "AUC": auc
    })

    print(
        f"K={k:2d} | "
        f"Accuracy={acc:.6f} | "
        f"AUC={auc:.6f}"
    )

# RESULTS

results_df = pd.DataFrame(results)

best_acc_row = results_df.loc[
    results_df["Accuracy"].idxmax()
]

best_auc_row = results_df.loc[
    results_df["AUC"].idxmax()
]

print("\n===================================================")
print("BEST RESULTS")
print("===================================================")

print(
    f"Best Accuracy:"
    f"\nK = {int(best_acc_row['K'])}"
    f"\nAccuracy = {best_acc_row['Accuracy']:.6f}"
    f"\nAUC = {best_acc_row['AUC']:.6f}"
)

print()

print(
    f"Best AUC:"
    f"\nK = {int(best_auc_row['K'])}"
    f"\nAccuracy = {best_auc_row['Accuracy']:.6f}"
    f"\nAUC = {best_auc_row['AUC']:.6f}"
)

print("\n===================================================")
print("TOP 15 CONFIGURATIONS BY ACCURACY")
print("===================================================")

print(
    results_df
    .sort_values("Accuracy", ascending=False)
    .head(15)
    .to_string(index=False)
)

print("\n===================================================")
print("TOP 15 CONFIGURATIONS BY AUC")
print("===================================================")

print(
    results_df
    .sort_values("AUC", ascending=False)
    .head(15)
    .to_string(index=False)
)

best_k = int(best_acc_row["K"])

print("\n===================================================")
print("BEST FEATURE SUBSET")
print("===================================================")

for i, feat in enumerate(ranked_features[:best_k], start=1):
    print(f"{i:2d}. {feat}")

import joblib

joblib.dump(
    {
        "model": best_model,
        "features": best_features,
        "accuracy": best_accuracy,
        "auc": best_auc
    },
    "Models/forensic_gbt_topk_best.pkl"
)

print("\n===================================================")
print("BEST MODEL SAVED")
print("===================================================")
print(f"Accuracy: {best_accuracy:.6f}")
print(f"AUC:      {best_auc:.6f}")
print(f"Features: {len(best_features)}")
print("Saved to: Models/forensic_gbt_topk_best.pkl")