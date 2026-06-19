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

# SEQUENTIAL FORWARD SELECTION

import joblib

print("SEQUENTIAL FORWARD SELECTION")

remaining_features = feature_cols.copy()
selected_features = []

results = []

best_model = None
best_score = -1

step = 1

while len(remaining_features) > 0:

    print("\n" + "=" * 60)
    print(
        f"STEP {step} | "
        f"Selected Features: {len(selected_features)}"
    )
    print("=" * 60)

    best_feature_this_round = None
    best_auc_this_round = -1
    best_acc_this_round = -1
    best_model_this_round = None

    for candidate in remaining_features:

        current_features = (
            selected_features + [candidate]
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
            X_train[current_features],
            y_train,
            eval_set=[
                (
                    X_train[current_features],
                    y_train
                ),
                (
                    X_val[current_features],
                    y_val
                )
            ],
            verbose=False
        )

        pred = model.predict(
            X_test[current_features]
        )

        prob = model.predict_proba(
            X_test[current_features]
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
            f"{candidate:<35} "
            f"AUC={auc:.6f} "
            f"ACC={acc:.6f}"
        )

        if auc > best_auc_this_round:

            best_auc_this_round = auc
            best_acc_this_round = acc

            best_feature_this_round = candidate
            best_model_this_round = model

    selected_features.append(
        best_feature_this_round
    )

    remaining_features.remove(
        best_feature_this_round
    )

    results.append({
        "step": step,
        "feature_added":
            best_feature_this_round,
        "num_features":
            len(selected_features),
        "auc":
            best_auc_this_round,
        "accuracy":
            best_acc_this_round
    })

    print()
    print(
        f"ADDED FEATURE: "
        f"{best_feature_this_round}"
    )

    print(
        f"AUC: "
        f"{best_auc_this_round:.6f}"
    )

    print(
        f"ACC: "
        f"{best_acc_this_round:.6f}"
    )

    if best_auc_this_round > best_score:

        best_score = best_auc_this_round

        best_model = best_model_this_round

        best_feature_snapshot = (
            selected_features.copy()
        )

    step += 1

# RESULTS

results_df = pd.DataFrame(results)

print("\n")
print("=" * 80)
print("FULL SFS RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)

best_row = results_df.loc[
    results_df["auc"].idxmax()
]

print("\n")
print("=" * 80)
print("BEST SFS RESULT")
print("=" * 80)

print(
    f"Features Used : "
    f"{best_row['num_features']}"
)

print(
    f"AUC           : "
    f"{best_row['auc']:.6f}"
)

print(
    f"Accuracy      : "
    f"{best_row['accuracy']:.6f}"
)

print("\n")
print("=" * 80)
print("BEST FEATURE SET")
print("=" * 80)

for idx, feat in enumerate(
        best_feature_snapshot,
        start=1
):
    print(
        f"{idx:2d}. {feat}"
    )

# SAVE BEST MODEL

joblib.dump(
    {
        "model": best_model,
        "features": best_feature_snapshot,
        "auc": best_row["auc"],
        "accuracy": best_row["accuracy"]
    },
    "Models/forensic_gbt_sfs_best_1.pkl"
)

print("\n")
print("=" * 80)
print("MODEL SAVED")
print("=" * 80)

print(
    "Models/forensic_gbt_sfs_best_1.pkl"
)