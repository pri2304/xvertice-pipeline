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

from itertools import combinations
import joblib

print("RUNNING TRIPLE SWEEP")

all_triples = list(
    combinations(feature_cols, 3)
)

print(
    f"Total triples: {len(all_triples)}"
)

results = []

best_model = None
best_triple = None
best_auc = -1
best_acc = -1

for idx, triple in enumerate(
        all_triples,
        start=1
):

    f1, f2, f3 = triple

    print(
        f"\n[{idx}/{len(all_triples)}] "
        f"{f1} + {f2}"
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

    selected_features = [f1, f2, f3]

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

    results.append({
        "Feature1": f1,
        "Feature2": f2,
        "Feature3": f3,
        "Accuracy": acc,
        "AUC": auc
    })

    print(
        f"AUC={auc:.6f} | "
        f"ACC={acc:.6f}"
    )

    if auc > best_auc:

        best_auc = auc
        best_acc = acc
        best_triple = [f1, f2, f3]
        best_model = model

# RESULTS

results_df = pd.DataFrame(results)

print("\n")
print("=" * 80)
print("TOP 25 TRIPLES BY AUC")
print("=" * 80)

print(
    results_df
    .sort_values(
        "AUC",
        ascending=False
    )
    .head(25)
    .to_string(index=False)
)

print("\n")
print("=" * 80)
print("TOP 25 TRIPLES BY ACCURACY")
print("=" * 80)

print(
    results_df
    .sort_values(
        "Accuracy",
        ascending=False
    )
    .head(25)
    .to_string(index=False)
)

print("\n")
print("=" * 80)
print("BEST TRIPLE")
print("=" * 80)

print(
    f"Feature 1 : "
    f"{best_triple[0]}"
)

print(
    f"Feature 2 : "
    f"{best_triple[1]}"
)

print(
    f"Feature 3 : "
    f"{best_triple[2]}"
)

print(
    f"AUC       : "
    f"{best_auc:.6f}"
)

print(
    f"ACC       : "
    f"{best_acc:.6f}"
)

# SAVE BEST MODEL

joblib.dump(
    {
        "model": best_model,
        "features": best_triple,
        "auc": best_auc,
        "accuracy": best_acc
    },
    "Models/forensic_gbt_triple_best.pkl"
)

print("\nSaved:")
print("Models/forensic_gbt_triple_best.pkl")
