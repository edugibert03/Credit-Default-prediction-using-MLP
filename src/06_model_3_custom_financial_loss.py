"""
06_MODEL_3_CUSTOM_FINANCIAL_LOSS.PY
===================================
Model 3 (Financial Optimization - Example-Dependent Cost-Sensitive Learning):
Multi-Layer Perceptron trained using a custom asymmetric loss function that directly
embeds the bank's cash flows into backpropagation:
  - False Negatives penalized by actual Realized Loss Given Default (LGD in €)
  - False Positives penalized by Expected Net Profit (Opportunity Cost in €)

Uses normalized 3-column target tensor [y_true, fn_cost, fp_cost], custom Keras metric
subclasses, initial output bias log(pos/neg), and a profit-maximizing threshold search
maximizing total Net Profit in Euros.
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.utils import shuffle
from utils_financial_evaluation import evaluate_financial_impact

tf.random.set_seed(42)
np.random.seed(42)

# Custom Asymmetric Financial Loss Function
def financial_loss(y_true_extended, y_pred):
    y_true  = y_true_extended[:, 0]  # 1 if default, 0 if paid
    fn_cost = y_true_extended[:, 1]  # Realized LGD / mean_loan
    fp_cost = y_true_extended[:, 2]  # Expected Net Profit / mean_loan

    # Numerical clipping to prevent log(0)
    y_pred = tf.clip_by_value(tf.squeeze(y_pred), keras.backend.epsilon(), 1.0 - keras.backend.epsilon())

    loss_fn = y_true * fn_cost * (-tf.math.log(y_pred))
    loss_fp = (1.0 - y_true) * fp_cost * (-tf.math.log(1.0 - y_pred))

    return tf.reduce_mean(loss_fn + loss_fp)

# Custom Metrics to slice column 0 of y_true_extended for Keras tracking
class TrueAccuracy(keras.metrics.BinaryAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, 0], y_pred, sample_weight)

class TrueAUC(keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, 0], y_pred, sample_weight)

class TruePrecision(keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, 0], y_pred, sample_weight)

class TrueRecall(keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, 0], y_pred, sample_weight)

def build_financial_mlp(input_dim: int, initial_bias_val: float):
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
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=financial_loss,
        metrics=[
            TrueAccuracy(name="accuracy"),
            TrueAUC(name="auc_roc"),
            TrueAUC(name="auc_pr", curve="PR"),
            TruePrecision(name="precision"),
            TrueRecall(name="recall")
        ]
    )
    return model

def main(data_dir: str, finance_dir: str):
    print("[+] Loading datasets for Model 3 (Custom Financial Loss)...")
    X_train_df = pd.read_parquet(os.path.join(data_dir, "X_TRAIN_FULL_8020.parquet"))
    X_test_df  = pd.read_parquet(os.path.join(data_dir, "X_TEST_FULL_8020.parquet"))
    Y_train_df = pd.read_parquet(os.path.join(data_dir, "Y_TRAIN_FULL_8020.parquet"))
    Y_test_df  = pd.read_parquet(os.path.join(data_dir, "Y_TEST_FULL_8020.parquet"))

    finance_train = pd.read_parquet(os.path.join(finance_dir, "FINANCE_TRAIN.parquet"))
    finance_test  = pd.read_parquet(os.path.join(finance_dir, "FINANCE_TEST.parquet"))

    X_train = X_train_df.to_numpy(dtype=np.float32)
    X_test  = X_test_df.to_numpy(dtype=np.float32)
    Y_train = Y_train_df.to_numpy(dtype=np.float32).ravel()
    Y_test  = Y_test_df.to_numpy(dtype=np.float32).ravel()

    # Normalization constant: average loan size to maintain stable gradients
    mean_loan = float(finance_train["loan_amnt"].mean())
    print(f"[+] Normalization constant (mean loan amount): €{mean_loan:,.2f}")

    # Pack 3-column target tensor [target, real_lgd_normalized, expected_profit_normalized]
    Y_train_extended = np.column_stack([
        Y_train.astype(int),
        finance_train["real_lgd"].values / mean_loan,
        finance_train["expected_profit"].values / mean_loan
    ]).astype("float32")

    Y_test_extended = np.column_stack([
        Y_test.astype(int),
        finance_test["real_lgd"].values / mean_loan,
        finance_test["expected_profit"].values / mean_loan
    ]).astype("float32")

    neg, pos = np.bincount(Y_train.astype(int))
    initial_bias_val = float(np.log(pos / neg))
    print(f"[+] Initial Bias b0: {initial_bias_val:.4f}")

    model = build_financial_mlp(input_dim=X_train.shape[1], initial_bias_val=initial_bias_val)
    model.summary()

    cb = [
        callbacks.EarlyStopping(monitor="val_auc_pr", mode="max", patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_auc_pr", mode="max", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    X_train_s, Y_train_ext_s = shuffle(X_train, Y_train_extended, random_state=42)

    print("[+] Training Model 3 (Instance-level Custom Financial Loss)...")
    model.fit(
        X_train_s, Y_train_ext_s,
        validation_split=0.15,
        epochs=100,
        batch_size=256,
        callbacks=cb,
        verbose=1
    )

    print("\n[+] Predicting on Test Set...")
    y_probs = model.predict(X_test, batch_size=512).ravel()
    y_true  = Y_test.astype(int)

    expected_profit = finance_test["expected_profit"].values
    real_lgd        = finance_test["real_lgd"].values

    # Grid search for threshold strictly maximizing Euro Net Profit
    print("[+] Searching for Profit-Maximizing Decision Threshold...")
    thresholds = np.arange(0.10, 0.90, 0.01)
    profits = []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        approved_paid    = (y_pred == 0) & (y_true == 0)
        approved_default = (y_pred == 0) & (y_true == 1)
        profit = np.sum(expected_profit[approved_paid]) - np.sum(real_lgd[approved_default])
        profits.append(profit)

    optimal_idx = np.argmax(profits)
    optimal_threshold = thresholds[optimal_idx]
    max_net_profit = profits[optimal_idx]
    print(f"[+] Optimal Profit Threshold: {optimal_threshold:.2f} (Projected Max Net Profit: €{max_net_profit:,.2f})")

    y_pred_optimal = (y_probs >= optimal_threshold).astype(int)

    evaluate_financial_impact(
        y_true=y_true,
        y_pred=y_pred_optimal,
        expected_profit=expected_profit,
        real_lgd=real_lgd,
        model_name="Model 3 — Custom Financial Loss (Profit-Driven EDCSL)"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model 3 (Custom Financial Loss)")
    parser.add_argument("--data_dir", default="data/full_8020")
    parser.add_argument("--finance_dir", default="data/finance")
    args = parser.parse_args()
    main(args.data_dir, args.finance_dir)
