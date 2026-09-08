"""
02A_PIPELINE_MICE_BALANCED.PY
=============================
Pipeline A: MICE Imputation, Random Undersampling (50/50 Balance),
Outlier/Negative Clipping, Robust Scaling, and Multicollinearity Filtering.

Produces the balanced dataset used by Model 1 (Baseline).
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import RobustScaler

def run_mice_balanced(input_dir: str, output_dir: str, random_seed: int = 42):
    print("[+] Loading clean split datasets...")
    X_train = pd.read_parquet(os.path.join(input_dir, "X_train_raw.parquet"))
    X_test  = pd.read_parquet(os.path.join(input_dir, "X_test_raw.parquet"))
    y_train = pd.read_parquet(os.path.join(input_dir, "y_train_raw.parquet")).iloc[:, 0]
    y_test  = pd.read_parquet(os.path.join(input_dir, "y_test_raw.parquet")).iloc[:, 0]

    # 1. MICE Imputation on numerical features (excluding addr_state)
    numerical_cols = X_train.columns.drop("addr_state")
    print(f"[+] Fitting MICE IterativeImputer on {len(numerical_cols)} numerical features...")
    imputer = IterativeImputer(max_iter=10, random_state=random_seed)
    X_train[numerical_cols] = imputer.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols]  = imputer.transform(X_test[numerical_cols])

    # 2. Clip negative values & extreme outliers
    for col in numerical_cols:
        if (X_train[col] < 0).any():
            X_train[col] = X_train[col].clip(lower=0.0)
            X_test[col]  = X_test[col].clip(lower=0.0)

    for col in ["percent_bc_gt_75", "pct_tl_nvr_dlq"]:
        if col in X_train.columns:
            X_train[col] = X_train[col].clip(upper=100.0)
            X_test[col]  = X_test[col].clip(upper=100.0)

    if "bc_util" in X_train.columns:
        X_train["bc_util"] = X_train["bc_util"].clip(upper=340.0)
        X_test["bc_util"]  = X_test["bc_util"].clip(upper=340.0)

    if "mths_since_recent_inq" in X_train.columns:
        X_train["mths_since_recent_inq"] = X_train["mths_since_recent_inq"].clip(upper=25.0)
        X_test["mths_since_recent_inq"]  = X_test["mths_since_recent_inq"].clip(upper=25.0)

    # 3. Class Balancing - RandomUnderSampler (50/50)
    print("[+] Applying RandomUnderSampler (50/50 balance)...")
    rus = RandomUnderSampler(sampling_strategy=1.0, random_state=random_seed)
    X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)

    # 4. Target Encoding for addr_state (calculated on pre-balanced training data)
    state_means = y_train.groupby(X_train["addr_state"]).mean()
    X_train_bal["addr_state"] = X_train_bal["addr_state"].map(state_means)
    X_test["addr_state"]      = X_test["addr_state"].map(state_means)

    # 5. RobustScaler
    print("[+] Scaling features with RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_bal), columns=X_train_bal.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # 6. Drop highly collinear features (Pearson |r| > 0.90)
    train_full = X_train_scaled.copy()
    train_full["TARGET"] = y_train_bal.values
    corr_matrix = X_train_scaled.corr().abs()
    target_corr = train_full.corr().abs()["TARGET"]
    upper_tri   = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper_tri.columns:
        high_corr = upper_tri.index[upper_tri[col] > 0.90].tolist()
        for c in high_corr:
            if target_corr[col] < target_corr[c]:
                to_drop.add(col)
            else:
                to_drop.add(c)

    # Drop zero variance dummy column if present
    for col in ["purpose_educational"]:
        if col in X_train_scaled.columns:
            to_drop.add(col)

    X_train_final = X_train_scaled.drop(columns=list(to_drop))
    X_test_final  = X_test_scaled.drop(columns=list(to_drop))
    print(f"[+] Dropped {len(to_drop)} collinear/zero-variance features. Remaining: {X_train_final.shape[1]}")

    os.makedirs(output_dir, exist_ok=True)
    X_train_final.to_parquet(os.path.join(output_dir, "X_TRAIN_MICE_BALANCED.parquet"), index=False)
    X_test_final.to_parquet(os.path.join(output_dir, "X_TEST_MICE_BALANCED.parquet"), index=False)
    y_train_bal.to_frame().to_parquet(os.path.join(output_dir, "Y_TRAIN_MICE_BALANCED.parquet"), index=False)
    y_test.to_frame().to_parquet(os.path.join(output_dir, "Y_TEST_MICE_BALANCED.parquet"), index=False)
    print(f"[✓] Successfully exported balanced datasets to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MICE + Balanced Pipeline")
    parser.add_argument("--input_dir", default="data/processed_raw")
    parser.add_argument("--output_dir", default="data/balanced_mice")
    args = parser.parse_args()
    run_mice_balanced(args.input_dir, args.output_dir)
