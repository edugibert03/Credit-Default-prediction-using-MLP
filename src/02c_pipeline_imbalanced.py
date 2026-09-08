"""
02C_PIPELINE_IMBALANCED.PY
==========================
Pipeline C: Production Full Dataset Pipeline (80/20 Natural Distribution).
Takes MICE-imputed datasets, omits RandomUnderSampler to preserve the ~80/20
natural loan distribution (~960,440 training records), applies Target Encoding,
RobustScaler, and feature selection.

Produces datasets for Model 2 (Class Weights) and Model 3 (Custom Financial Loss).
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler

def run_imbalanced_pipeline(input_dir: str, output_dir: str, random_seed: int = 42):
    print("[+] Loading raw clean split datasets...")
    X_train = pd.read_parquet(os.path.join(input_dir, "X_train_raw.parquet"))
    X_test  = pd.read_parquet(os.path.join(input_dir, "X_test_raw.parquet"))
    y_train = pd.read_parquet(os.path.join(input_dir, "y_train_raw.parquet")).iloc[:, 0]
    y_test  = pd.read_parquet(os.path.join(input_dir, "y_test_raw.parquet")).iloc[:, 0]

    # 1. MICE Imputation on numerical features
    numerical_cols = X_train.columns.drop("addr_state")
    print(f"[+] Fitting MICE on full imbalanced dataset ({len(X_train):,} training records)...")
    imputer = IterativeImputer(max_iter=10, random_state=random_seed)
    X_train[numerical_cols] = imputer.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols]  = imputer.transform(X_test[numerical_cols])

    # 2. Clipping
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

    # 3. NO UNDERSAMPLING — PRESERVE NATURAL CLASS RATIO (~80% Paid, ~20% Default)
    print("[+] Preserving natural class imbalance (~80/20 distribution).")

    # 4. Target Encoding for addr_state on full training set
    state_means = y_train.groupby(X_train["addr_state"]).mean()
    X_train["addr_state"] = X_train["addr_state"].map(state_means)
    X_test["addr_state"]  = X_test["addr_state"].map(state_means)

    # 5. RobustScaler
    print("[+] Scaling features with RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # 6. Collinearity Drop (|r| > 0.90)
    train_full = X_train_scaled.copy()
    train_full["TARGET"] = y_train.values
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

    for col in ["purpose_educational"]:
        if col in X_train_scaled.columns:
            to_drop.add(col)

    X_train_final = X_train_scaled.drop(columns=list(to_drop))
    X_test_final  = X_test_scaled.drop(columns=list(to_drop))
    print(f"[+] Final full dataset features: {X_train_final.shape[1]}")

    os.makedirs(output_dir, exist_ok=True)
    X_train_final.to_parquet(os.path.join(output_dir, "X_TRAIN_FULL_8020.parquet"), index=False)
    X_test_final.to_parquet(os.path.join(output_dir, "X_TEST_FULL_8020.parquet"), index=False)
    y_train.to_frame().to_parquet(os.path.join(output_dir, "Y_TRAIN_FULL_8020.parquet"), index=False)
    y_test.to_frame().to_parquet(os.path.join(output_dir, "Y_TEST_FULL_8020.parquet"), index=False)
    print(f"[✓] Successfully exported full 80/20 datasets to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full 80/20 Dataset Pipeline")
    parser.add_argument("--input_dir", default="data/processed_raw")
    parser.add_argument("--output_dir", default="data/full_8020")
    args = parser.parse_args()
    run_imbalanced_pipeline(args.input_dir, args.output_dir)
