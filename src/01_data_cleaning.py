"""
01_DATA_CLEANING.PY
===================
Complete data cleaning, feature leakage removal, and categorical feature engineering
pipeline for the Lending Club loan dataset (1.3M+ resolved loans).

Extracts resolved loans (Fully Paid vs. Charged Off), removes forward-looking data leakage,
encodes categorical features, and performs initial stratified 80/20 train/test split.
"""

import os
import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

def clean_data(input_file: str, output_dir: str, random_seed: int = 42):
    print(f"[+] Loading raw dataset from: {input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"    Initial shape: {df.shape}")

    # 1. Filter resolved loans & create binary target (0: Fully Paid, 1: Charged Off)
    resolved_statuses = ["Fully Paid", "Charged Off"]
    df = df[df["loan_status"].isin(resolved_statuses)].copy()
    df["target"] = (df["loan_status"] == "Charged Off").astype(int)
    df.drop(columns=["loan_status"], inplace=True)
    print(f"[+] Filtered completed loans: {len(df):,} records")

    # 2. Drop columns with 100% missing values
    missing_pct = df.isnull().mean() * 100
    df.drop(columns=missing_pct[missing_pct == 100].index.tolist(), inplace=True)

    # 3. Drop constant / single-value columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=constant_cols, inplace=True)

    # 4. Drop data-leakage columns (post-approval, post-issuance, and settlement)
    hardship_cols = [col for col in df.columns if "hardship" in col.lower() or "settlement" in col.lower()]
    df.drop(columns=hardship_cols, inplace=True)

    time_after_loan_cols = [
        "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
        "total_rec_late_fee", "recoveries", "collection_recovery_fee",
        "last_pymnt_d", "last_pymnt_amnt", "last_credit_pull_d",
        "last_fico_range_high", "last_fico_range_low"
    ]
    df.drop(columns=time_after_loan_cols, inplace=True, errors="ignore")

    approval_cols = ["funded_amnt", "funded_amnt_inv", "url", "initial_list_status", "disbursement_method"]
    df.drop(columns=approval_cols, inplace=True, errors="ignore")

    # 5. Drop identifier and free-text columns
    id_text_cols = ["id", "emp_title", "title", "zip_code", "desc"]
    df.drop(columns=id_text_cols, inplace=True, errors="ignore")

    # 6. Drop columns with >40% missing values
    high_missing = df.columns[(df.isnull().sum() / len(df) * 100) > 40].tolist()
    df.drop(columns=high_missing, inplace=True)
    print(f"    Columns after dropping missing/leakage: {df.shape[1]}")

    # 7. Categorical Variable Encoding
    # 7a. term: strip to integer months (36, 60)
    df["term"] = df["term"].astype(str).str.strip().str.replace("months", "").astype(int)

    # 7b. emp_length: drop nulls (~6%), map ordinal integers (0-10)
    df.dropna(subset=["emp_length"], inplace=True)
    emp_map = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
        "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10
    }
    df["emp_length"] = df["emp_length"].map(emp_map)

    # 7c. sub_grade: ordinal encoding 1-35 (drops redundant grade)
    df.drop(columns=["grade"], inplace=True, errors="ignore")
    sub_grades = [f"{g}{n}" for g in ["A","B","C","D","E","F","G"] for n in range(1, 6)]
    sub_grade_map = {sg: i + 1 for i, sg in enumerate(sub_grades)}
    df["sub_grade"] = df["sub_grade"].map(sub_grade_map)

    # 7d. One-Hot Encoding for categorical features
    df = pd.get_dummies(df, columns=["home_ownership", "verification_status", "purpose"], drop_first=True, dtype=int)

    # 7e. Earliest credit line to duration in months relative to issue date
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"])
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    years_diff = (df["issue_d"].dt.year - df["earliest_cr_line"].dt.year) * 12
    months_diff = df["issue_d"].dt.month - df["earliest_cr_line"].dt.month
    df["mnths_since_earliest_cr"] = years_diff + months_diff
    df.drop(columns=["earliest_cr_line", "issue_d"], inplace=True)

    # 7f. Application type
    df["application_type"] = df["application_type"].map({"Individual": 0, "Joint App": 1})

    # 8. Row Filtering: drop low-missing (<1%) and excessive missing (>6 nulls per row)
    null_pct = df.isnull().mean() * 100
    low_null_cols = null_pct[(null_pct > 0) & (null_pct <= 1)].index.tolist()
    df.dropna(subset=low_null_cols, inplace=True)
    df = df[df.isnull().sum(axis=1) <= 6].copy()

    # 9. Stratified Train/Test Split (80/20)
    print(f"[+] Cleaned dataset: {df.shape[0]:,} rows x {df.shape[1]} features")
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_seed, stratify=y
    )
    print(f"[+] Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    os.makedirs(output_dir, exist_ok=True)
    X_train.to_parquet(os.path.join(output_dir, "X_train_raw.parquet"), index=False)
    X_test.to_parquet(os.path.join(output_dir, "X_test_raw.parquet"), index=False)
    y_train.to_frame().to_parquet(os.path.join(output_dir, "y_train_raw.parquet"), index=False)
    y_test.to_frame().to_parquet(os.path.join(output_dir, "y_test_raw.parquet"), index=False)
    print(f"[✓] Saved raw split datasets to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Cleaning & Preprocessing Pipeline")
    parser.add_argument("--input", default="data/accepted_2007_to_2018Q4.csv", help="Path to raw Lending Club CSV")
    parser.add_argument("--output", default="data/processed_raw", help="Output directory")
    args = parser.parse_args()
    clean_data(args.input, args.output)
