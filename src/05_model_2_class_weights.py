"""
05_MODEL_2_CLASS_WEIGHTS.PY
===========================
Model 2 (Cost-Sensitive Learning - Class Weights): Multi-Layer Perceptron trained
on the full natural imbalanced dataset (~80% paid, ~20% default).
Uses calculated balanced class weights (~4:1 penalty on defaults), initial output bias
b0 = log(pos/neg), and monitors val_auc_pr for early stopping.
Decision threshold selected via F1-Score optimization.
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve
from utils_financial_evaluation import evaluate_financial_impact

tf.random.set_seed(42)
np.random.seed(42)

def build_weighted_mlp(input_dim: int, initial_bias_val: float):
    initial_bias = tf.keras.initializers.Constant(value=initial_bias_val)
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(256, kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),
        layers.Dense(128, kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),
        layers.Dense(64, kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid", bias_initializer=initial_bias)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc_roc"),
            keras.metrics.AUC(name="auc_pr", curve="PR"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )
    return model

def main(data_dir: str, finance_dir: str):
    print("[+] Loading full 80/20 datasets for Model 2...")
    X_train_df = pd.read_parquet(os.path.join(data_dir, "X_TRAIN_FULL_8020.parquet"))
    X_test_df  = pd.read_parquet(os.path.join(data_dir, "X_TEST_FULL_8020.parquet"))
    Y_train_df = pd.read_parquet(os.path.join(data_dir, "Y_TRAIN_FULL_8020.parquet"))
    Y_test_df  = pd.read_parquet(os.path.join(data_dir, "Y_TEST_FULL_8020.parquet"))

    finance_test = pd.read_parquet(os.path.join(finance_dir, "FINANCE_TEST.parquet"))

    X_train = X_train_df.to_numpy(dtype=np.float32)
    X_test  = X_test_df.to_numpy(dtype=np.float32)
    Y_train = Y_train_df.to_numpy(dtype=np.float32).ravel()
    Y_test  = Y_test_df.to_numpy(dtype=np.float32).ravel()

    # Calculate class weights
    classes = np.unique(Y_train)
    weights = compute_class_weight("balanced", classes=classes, y=Y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"[+] Computed Class Weights: {class_weights} (Ratio: {class_weights[1]/class_weights[0]:.2f}x)")

    # Output bias initialization
    neg, pos = np.bincount(Y_train.astype(int))
    initial_bias_val = float(np.log(pos / neg))
    print(f"[+] Output Bias Initializer b0 = log(pos/neg) = {initial_bias_val:.4f}")

    model = build_weighted_mlp(input_dim=X_train.shape[1], initial_bias_val=initial_bias_val)
    model.summary()

    cb = [
        callbacks.EarlyStopping(monitor="val_auc_pr", mode="max", patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_auc_pr", mode="max", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    X_train_s, Y_train_s = shuffle(X_train, Y_train, random_state=42)

    print("[+] Training Model 2 (Class-Weighted Cost-Sensitive)...")
    model.fit(
        X_train_s, Y_train_s,
        validation_split=0.15,
        epochs=100,
        batch_size=256,
        class_weight=class_weights,
        callbacks=cb,
        verbose=1
    )

    print("\n[+] Evaluating on Test Set...")
    eval_results = model.evaluate(X_test, Y_test, verbose=0)
    print(f"    Loss:      {eval_results[0]:.4f}")
    print(f"    Accuracy:  {eval_results[1]:.4f}")
    print(f"    AUC-ROC:   {eval_results[2]:.4f}")
    print(f"    AUC-PR:    {eval_results[3]:.4f}")

    y_probs = model.predict(X_test, batch_size=512).ravel()
    precisions, recalls, thresholds = precision_recall_curve(Y_test.astype(int), y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"[+] Optimal F1 Decision Threshold: {optimal_threshold:.4f}")

    y_pred = (y_probs >= optimal_threshold).astype(int)

    evaluate_financial_impact(
        y_true=Y_test,
        y_pred=y_pred,
        expected_profit=finance_test["expected_profit"].values,
        real_lgd=finance_test["real_lgd"].values,
        model_name="Model 2 — Class Weights"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model 2 (Class Weights)")
    parser.add_argument("--data_dir", default="data/full_8020")
    parser.add_argument("--finance_dir", default="data/finance")
    args = parser.parse_args()
    main(args.data_dir, args.finance_dir)
