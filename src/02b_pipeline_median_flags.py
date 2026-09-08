"""
02B_PIPELINE_MEDIAN_FLAGS.PY
============================
Pipeline B: Median Imputation with Binary Missingness Flags.
Serves as the imputation benchmark in Phase I to compare against MICE.
"""

import os
import argparse
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import RobustScaler

def run_median_pipeline(input_dir: str, output_dir: str, random_seed: int = 42):
    print("[+] Loading clean split datasets...")
    X_train = pd.read_parquet(os.path.join(input_dir, "X_train_raw.parquet"))
    X_test  = pd.read_parquet(os.path.join(input_dir, "X_test_raw.parquet"))
    y_train = pd.read_parquet(os.path.join(input_dir, "y_train_raw.parquet")).iloc[:, 0]
    y_test  = pd.read_parquet(os.path.join(input_dir, "y_test_raw.parquet")).iloc[:, 0]

    numerical_cols = X_train.columns.drop("addr_state")
    cols_with_nulls = [c for c in numerical_cols if X_train[c].isnull().any() or X_test[c].isnull().any()]

    print(f"[+] Creating binary missingness flags for {len(cols_with_nulls)} columns...")
    for c in cols_with_nulls:
        X_train[f"{c}_missing"] = X_train[c].isnull().astype(int)
        X_test[f"{c}_missing"]  = X_test[c].isnull().astype(int)

    train_medians = X_train[cols_with_nulls].median()
    for c in cols_with_nulls:
        X_train[c] = X_train[c].fillna(train_medians[c])
        X_test[c]  = X_test[c].fillna(train_medians[c])

    # Balancing 50/50
    rus = RandomUnderSampler(sampling_strategy=1.0, random_state=random_seed)
    X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)

    state_means = y_train.groupby(X_train["addr_state"]).mean()
    X_train_bal["addr_state"] = X_train_bal["addr_state"].map(state_means)
    X_test["addr_state"]      = X_test["addr_state"].map(state_means)

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_bal), columns=X_train_bal.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Collinearity filter
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

    for col in ["purpose_educational"]:
        if col in X_train_scaled.columns:
            to_drop.add(col)

    X_train_final = X_train_scaled.drop(columns=list(to_drop))
    X_test_final  = X_test_scaled.drop(columns=list(to_drop))

    os.makedirs(output_dir, exist_ok=True)
    X_train_final.to_parquet(os.path.join(output_dir, "X_TRAIN_MEDIAN.parquet"), index=False)
    X_test_final.to_parquet(os.path.join(output_dir, "X_TEST_MEDIAN.parquet"), index=False)
    y_train_bal.to_frame().to_parquet(os.path.join(output_dir, "Y_TRAIN_MEDIAN.parquet"), index=False)
    y_test.to_frame().to_parquet(os.path.join(output_dir, "Y_TEST_MEDIAN.parquet"), index=False)
    print(f"[✓] Successfully exported median benchmark datasets to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Median Imputation Benchmark Pipeline")
    parser.add_argument("--input_dir", default="data/processed_raw")
    parser.add_argument("--output_dir", default="data/median_benchmark")
    args = parser.parse_args()
    run_median_pipeline(args.input_dir, args.output_dir)
