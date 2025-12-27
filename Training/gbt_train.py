import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve, average_precision_score)
import joblib
import os

# ================= CONFIGURATION =================
INPUT_CSV = "gbt_training_features_flattened.csv"
MODEL_PKL_PATH = "../Models/forensic_gbt_model.pkl"
FEATURE_CSV_PATH = "../GBT Stuff/feature_importance_full.csv"  # <-- NEW: Saves full list here
TEST_SIZE = 0.15
VAL_SIZE = 0.15


# =================================================

def train_forensic_gbt():
    print("--- Loading Dataset ---")
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} samples.")

    # 1. SETUP FEATURES
    ignore_cols = ['path', 'label_str', 'tag1', 'time', 'label']
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feature_cols]
    y = df['label']
    tags = df['tag1']

    # 2. STRATIFIED SPLIT
    print("\n--- Splitting Dataset ---")
    try:
        X_temp, X_test, y_temp, y_test, tags_temp, tags_test = train_test_split(
            X, y, tags, test_size=TEST_SIZE, stratify=tags, random_state=42
        )
        val_split_adjusted = VAL_SIZE / (1 - TEST_SIZE)
        X_train, X_val, y_train, y_val, tags_train, tags_val = train_test_split(
            X_temp, y_temp, tags_temp, test_size=val_split_adjusted, stratify=tags_temp, random_state=42
        )
    except ValueError:
        print("⚠️ Tag stratification failed. Falling back to simple stratification.")
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=42)
        val_split_adjusted = VAL_SIZE / (1 - TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_split_adjusted, stratify=y_temp,
                                                          random_state=42)
        tags_test = tags.loc[X_test.index]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 3. TRAIN XGBOOST
    print("\n--- Training GBT ---")
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
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )

    # 4. PREDICTIONS
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ================= FULL FEATURE IMPORTANCE =================
    print("\n" + "=" * 40)
    print("FULL FEATURE IMPORTANCE RANKING")
    print("=" * 40)

    importance = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})

    # Sort by importance (Highest first)
    feat_df = feat_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # Save to CSV
    feat_df.to_csv(FEATURE_CSV_PATH, index=False)
    print(f"✅ Full list saved to {FEATURE_CSV_PATH}")

    # Print FULL list to terminal
    pd.set_option('display.max_rows', None)  # Force pandas to print all rows
    print(feat_df)
    pd.reset_option('display.max_rows')  # Reset to default

    # ================= VISUALIZATION BLOCK =================
    print("\n--- Generating Graphs ---")

    # Graph A: Feature Importance (Top 25)
    plt.figure(figsize=(10, 10))
    sns.barplot(x='Importance', y='Feature', data=feat_df.head(25), palette="viridis")
    plt.title("Top 25 Forensic Features")
    plt.tight_layout()
    plt.savefig("graph_feature_importance.png")
    plt.close()

    # Graph B: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("graph_confusion_matrix.png")
    plt.close()

    # Graph C: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("graph_roc_curve.png")
    plt.close()

    # Graph D: Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig("graph_pr_curve.png")
    plt.close()

    # Graph E: Training History
    results = model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
    plt.plot(x_axis, results['validation_1']['logloss'], label='Validation')
    plt.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Training Loss')
    plt.savefig("graph_training_loss.png")
    plt.close()

    print("✅ All 5 graphs saved.")

    # ================= DETAILED METRICS =================
    print("\n" + "=" * 40)
    print("DETAILED METRICS")
    print("=" * 40)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    fpr_rate = fp / (tn + fp)
    fnr_rate = fn / (tp + fn)

    print(f"Accuracy:    {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC Score:   {roc_auc:.4f}")
    print("-" * 20)
    print(f"Sensitivity (Recall on Fakes): {sensitivity:.4f}")
    print(f"Specificity (Recall on Real):  {specificity:.4f}")
    print("-" * 20)
    print(f"False Positive Rate (Real flagged as Fake): {fpr_rate:.4f}")
    print(f"False Negative Rate (Fake flagged as Real): {fnr_rate:.4f}")

    # ================= TAG ANALYSIS =================
    print("\n--- Accuracy by Generator/Tag ---")
    results_df = pd.DataFrame({'Tag': tags_test, 'True': y_test, 'Pred': y_pred})
    results_df['Correct'] = results_df['True'] == results_df['Pred']

    tag_stats = results_df.groupby('Tag').agg(
        Count=('Correct', 'count'),
        Accuracy=('Correct', 'mean'),
        FalsePos=('Correct', lambda x: (~x & (results_df.loc[x.index, 'True'] == 0)).sum()),
        FalseNeg=('Correct', lambda x: (~x & (results_df.loc[x.index, 'True'] == 1)).sum())
    ).sort_values(by='Accuracy', ascending=False)

    print(f"{'TAG':<30} | {'COUNT':<6} | {'ACCURACY':<9} | {'FP':<4} | {'FN':<4}")
    print("-" * 65)
    for tag, row in tag_stats.iterrows():
        acc_str = f"{row['Accuracy'] * 100:.1f}%"
        print(f"{tag:<30} | {row['Count']:<6} | {acc_str:<9} | {int(row['FalsePos']):<4} | {int(row['FalseNeg']):<4}")

    # 6. SAVE MODEL
    joblib.dump(model, MODEL_PKL_PATH)
    print(f"\n✅ Model saved to {MODEL_PKL_PATH}")


if __name__ == "__main__":
    train_forensic_gbt()