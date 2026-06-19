import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# CONFIG

INPUT_CSV = "/home/pri/PycharmProjects/forensicspipeline/Datasets/final_merged_dataset.csv"

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

feature_cols = [
    c for c in df.columns
    if c not in ignore_cols
]

print(f"Feature count: {len(feature_cols)}")

X = df[feature_cols]
y = df["label"]
tags = df["tag1"]

# SPLIT

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

    print("Tag stratification failed.")

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

full_acc = accuracy_score(
    y_test,
    full_pred
)

full_auc = roc_auc_score(
    y_test,
    full_prob
)

print(f"Full Accuracy: {full_acc:.6f}")
print(f"Full AUC:      {full_auc:.6f}")

# FEATURE IMPORTANCE

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": full_model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

ranked_features = (
    importance_df["Feature"]
    .tolist()
)

# ABLATION SWEEP

print("RUNNING ABLATION SWEEP")

results = []

worst_drop = -999
most_important_feature = None

best_model = None
best_features = None
best_auc = -1

for idx, removed_feature in enumerate(
        ranked_features,
        start=1
):

    print(
        f"\n[{idx}/{len(ranked_features)}] "
        f"Removing: {removed_feature}"
    )

    selected_features = [
        f for f in feature_cols
        if f != removed_feature
    ]

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
            (
                X_train[selected_features],
                y_train
            ),
            (
                X_val[selected_features],
                y_val
            )
        ],
        verbose=False
    )

    pred = model.predict(
        X_test[selected_features]
    )

    prob = model.predict_proba(
        X_test[selected_features]
    )[:, 1]

    acc = accuracy_score(
        y_test,
        pred
    )

    auc = roc_auc_score(
        y_test,
        prob
    )

    auc_drop = full_auc - auc
    acc_drop = full_acc - acc

    results.append({
        "Removed_Feature":
            removed_feature,
        "Accuracy":
            acc,
        "AUC":
            auc,
        "Accuracy_Drop":
            acc_drop,
        "AUC_Drop":
            auc_drop
    })

    print(
        f"AUC={auc:.6f} "
        f"(Drop={auc_drop:.6f})"
    )

    if auc_drop > worst_drop:

        worst_drop = auc_drop
        most_important_feature = (
            removed_feature
        )

    if auc > best_auc:

        best_auc = auc
        best_model = model
        best_features = (
            selected_features.copy()
        )

# RESULTS

results_df = pd.DataFrame(results)

print("\n")
print("=" * 80)
print("MOST IMPORTANT FEATURES")
print("=" * 80)

print(
    results_df
    .sort_values(
        "AUC_Drop",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

print("\n")
print("=" * 80)
print("LEAST IMPORTANT FEATURES")
print("=" * 80)

print(
    results_df
    .sort_values(
        "AUC_Drop",
        ascending=True
    )
    .head(20)
    .to_string(index=False)
)

print("\n")
print("=" * 80)
print("MOST CRITICAL FEATURE")
print("=" * 80)

print(
    f"Feature: {most_important_feature}"
)

print(
    f"AUC Drop: {worst_drop:.6f}"
)

# SAVE BEST MODEL

joblib.dump(
    {
        "model": best_model,
        "features": best_features,
        "auc": best_auc
    },
    "/home/pri/PycharmProjects/forensicspipeline/Models/forensic_gbt_ablation_best.pkl"
)

results_df.to_csv(
    "ablation_sweep_results.csv",
    index=False
)

print("\nSaved:")
print("Models/forensic_gbt_ablation_best.pkl")
print("ablation_sweep_results.csv")