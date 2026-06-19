import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import joblib
import os

# CONFIGURATION
INPUT_CSV = "Datasets/final_merged_dataset.csv"

MODEL_PKL_PATH = "Models/forensic_gbt_filtered_tags.pkl"

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

def train_forensic_gbt():
    print("--- Loading Dataset ---")

    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} samples.")

    # FILTER EXCLUDED TAGS
    before_count = len(df)

    df = df[~df["tag1"].isin(REMOVE_TAGS)].copy()

    removed_count = before_count - len(df)

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 - FULL MODEL (FILTERED DATASET)")
    print("=" * 60)
    print(f"Removed samples: {removed_count}")
    print(f"Remaining samples: {len(df)}")
    print("=" * 60)

    # SETUP FEATURES
    ignore_cols = ['path', 'label_str', 'tag1', 'time', 'label']
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feature_cols]
    y = df['label']
    tags = df['tag1']

    # STRATIFIED SPLIT
    print("\n--- Splitting Dataset ---")

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

        X_train, X_val, y_train, y_val, tags_train, tags_val = train_test_split(
            X_temp,
            y_temp,
            tags_temp,
            test_size=val_split_adjusted,
            stratify=tags_temp,
            random_state=42
        )

    except ValueError:
        print("⚠️ Tag stratification failed. Falling back to label stratification.")

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

        tags_test = tags.loc[X_test.index]

    print(f"Train: {len(X_train)}")
    print(f"Val:   {len(X_val)}")
    print(f"Test:  {len(X_test)}")

    # TRAIN MODEL
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
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_val, y_val)
        ],
        verbose=100
    )

    # PREDICTIONS
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # FEATURE IMPORTANCE
    print("\n" + "=" * 40)
    print("FULL FEATURE IMPORTANCE RANKING")
    print("=" * 40)

    importance = model.feature_importances_

    feat_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    })

    feat_df = feat_df.sort_values(
        by='Importance',
        ascending=False
    ).reset_index(drop=True)

    print(f"✅ Feature importance saved to:")

    pd.set_option('display.max_rows', None)
    print(feat_df)
    pd.reset_option('display.max_rows')

    # GRAPHS
    print("\n--- Generating Graphs ---")

    # Feature Importance
    plt.figure(figsize=(10, 10))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feat_df.head(25),
        palette="viridis"
    )
    plt.title("Top 25 Forensic Features")
    plt.tight_layout()
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake']
    )

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.4f})'
    )
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.close()

    # Precision Recall
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        lw=2,
        label=f'PR Curve (AP = {pr_auc:.4f})'
    )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc="lower left")
    plt.close()

    # Training Loss
    results = model.evals_result()

    epochs = len(results['validation_0']['logloss'])
    x_axis = range(epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_axis,
        results['validation_0']['logloss'],
        label='Train'
    )
    plt.plot(
        x_axis,
        results['validation_1']['logloss'],
        label='Validation'
    )

    plt.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Training Loss')
    plt.close()

    print("✅ All graphs saved.")

    # METRICS
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

    print(f"False Positive Rate: {fpr_rate:.4f}")
    print(f"False Negative Rate: {fnr_rate:.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    # TAG ANALYSIS
    print("\n--- Accuracy by Generator/Tag ---")

    results_df = pd.DataFrame({
        'Tag': tags_test,
        'True': y_test,
        'Pred': y_pred
    })

    results_df['Correct'] = (
        results_df['True'] == results_df['Pred']
    )

    tag_stats = results_df.groupby('Tag').agg(
        Count=('Correct', 'count'),
        Accuracy=('Correct', 'mean'),
        FalsePos=(
            'Correct',
            lambda x: (
                ~x &
                (results_df.loc[x.index, 'True'] == 0)
            ).sum()
        ),
        FalseNeg=(
            'Correct',
            lambda x: (
                ~x &
                (results_df.loc[x.index, 'True'] == 1)
            ).sum()
        )
    ).sort_values(
        by='Accuracy',
        ascending=False
    )

    print(
        f"{'TAG':<30} | "
        f"{'COUNT':<6} | "
        f"{'ACCURACY':<9} | "
        f"{'FP':<4} | "
        f"{'FN':<4}"
    )

    print("-" * 65)

    for tag, row in tag_stats.iterrows():
        acc_str = f"{row['Accuracy'] * 100:.1f}%"

        print(
            f"{tag:<30} | "
            f"{row['Count']:<6} | "
            f"{acc_str:<9} | "
            f"{int(row['FalsePos']):<4} | "
            f"{int(row['FalseNeg']):<4}"
        )

    # SAVE MODEL
    joblib.dump(model, MODEL_PKL_PATH)

    print(f"\n✅ Model saved to:")
    print(MODEL_PKL_PATH)


if __name__ == "__main__":
    train_forensic_gbt()