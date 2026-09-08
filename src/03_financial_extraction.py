"""
03_FINANCIAL_EXTRACTION.PY
==========================
Extraction of financial cash flows for the credit portfolio (French Amortization System,
Cost of Funds 3%, Expected Net Profit, and Realized Loss Given Default).

Constructs the financial side-channel matrices (FINANCE_TRAIN and FINANCE_TEST) aligned
row-for-row with the predictive feature datasets to enable monetary evaluation and
example-dependent cost-sensitive optimization.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def extract_financial_metrics(raw_csv_path: str, raw_processed_dir: str, output_dir: str, random_seed: int = 42):
    print(f"[+] Loading financial columns from raw CSV: {raw_csv_path}")
    financial_cols = [
        "loan_amnt", "int_rate", "term", "total_pymnt",
        "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
        "recoveries", "collection_recovery_fee"
    ]
    df_raw = pd.read_csv(raw_csv_path, low_memory=False, usecols=financial_cols + ["loan_status"])

    # Match filtering from 01_data_cleaning
    resolved_statuses = ["Fully Paid", "Charged Off"]
    df_raw = df_raw[df_raw["loan_status"].isin(resolved_statuses)].copy()
    df_raw["target"] = (df_raw["loan_status"] == "Charged Off").astype(int)

    # Clean term to integer
    df_raw["term"] = df_raw["term"].astype(str).str.strip().str.replace("months", "").astype(int)

    print("[+] Calculating French Amortization & banking financial metrics...")
    finance_df = pd.DataFrame(index=df_raw.index)
    finance_df["loan_amnt"] = df_raw["loan_amnt"]
    finance_df["int_rate"]  = df_raw["int_rate"]
    finance_df["term"]      = df_raw["term"]

    # Monthly interest rate and duration
    r = (df_raw["int_rate"] / 100.0) / 12.0
    n = df_raw["term"]

    # French Amortization installment PMT
    pmt = df_raw["loan_amnt"] * (r * (1.0 + r)**n) / ((1.0 + r)**n - 1.0)
    expected_gross_interest = (pmt * n) - df_raw["loan_amnt"]

    # Bank Cost of Funds (assumed 3.0% annual cost of capital)
    COST_OF_FUNDS_RATE = 0.03
    c = COST_OF_FUNDS_RATE / 12.0
    cof_pmt = df_raw["loan_amnt"] * (c * (1.0 + c)**n) / ((1.0 + c)**n - 1.0)
    expected_cof = (cof_pmt * n) - df_raw["loan_amnt"]

    # Expected Net Profit (earned if loan is fully paid)
    finance_df["expected_profit"] = expected_gross_interest - expected_cof

    # Realized Loss Given Default (LGD) from actual cash flows
    cash_in = (
        df_raw["total_rec_prncp"] +
        df_raw["total_rec_int"] +
        df_raw["total_rec_late_fee"] +
        (df_raw["recoveries"] - df_raw["collection_recovery_fee"])
    )
    months_active = np.clip(df_raw["total_pymnt"] / pmt, 0, n)
    cof_paid = expected_cof * (months_active / n)
    cash_out = df_raw["loan_amnt"] + cof_paid

    # Real LGD cannot be negative (if bank recovered more than principal+CoF, loss is 0)
    finance_df["real_lgd"] = np.clip(cash_out - cash_in, 0.0, None)

    print(f"[+] Average Loan Amount: €{finance_df['loan_amnt'].mean():,.2f}")
    print(f"[+] Average Expected Net Profit: €{finance_df['expected_profit'].mean():,.2f}")
    print(f"[+] Average Realized LGD (Defaults): €{finance_df.loc[df_raw['target']==1, 'real_lgd'].mean():,.2f}")

    # Synchronized 80/20 train/test split matching random state and stratification
    y = df_raw["target"]
    finance_train, finance_test, _, _ = train_test_split(
        finance_df, y, test_size=0.20, random_state=random_seed, stratify=y
    )

    os.makedirs(output_dir, exist_ok=True)
    finance_train.to_parquet(os.path.join(output_dir, "FINANCE_TRAIN.parquet"), index=False)
    finance_test.to_parquet(os.path.join(output_dir, "FINANCE_TEST.parquet"), index=False)
    finance_train.to_csv(os.path.join(output_dir, "FINANCE_TRAIN.csv"), index=False)
    finance_test.to_csv(os.path.join(output_dir, "FINANCE_TEST.csv"), index=False)
    print(f"[✓] Saved financial context datasets to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial Context Extraction Pipeline")
    parser.add_argument("--raw_csv", default="data/accepted_2007_to_2018Q4.csv")
    parser.add_argument("--processed_dir", default="data/processed_raw")
    parser.add_argument("--output_dir", default="data/finance")
    args = parser.parse_args()
    extract_financial_metrics(args.raw_csv, args.processed_dir, args.output_dir)
