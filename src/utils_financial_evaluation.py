"""
UTILS_FINANCIAL_EVALUATION.PY
=============================
Shared banking economics evaluation module for credit default prediction models.
Computes the Financial Confusion Matrix (TN Profit, FN Realized LGD, FP Opportunity Cost,
TP Losses Avoided) and overall Net Profit in Euros under realistic banking P&L accounting.
"""

import numpy as np

def evaluate_financial_impact(y_true: np.ndarray, y_pred: np.ndarray, 
                              expected_profit: np.ndarray, real_lgd: np.ndarray,
                              model_name: str = "Model") -> dict:
    """
    Computes the Euro P&L outcome based on actual credit cash flows.
    
    y_true: 0 (Fully Paid), 1 (Charged Off)
    y_pred: 0 (Approved), 1 (Rejected)
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    expected_profit = np.asarray(expected_profit).ravel()
    real_lgd = np.asarray(real_lgd).ravel()

    # Confusion matrix masks
    approved_paid      = (y_pred == 0) & (y_true == 0)  # True Negative (TN)
    approved_defaulted = (y_pred == 0) & (y_true == 1)  # False Negative (FN)
    rejected_paid      = (y_pred == 1) & (y_true == 0)  # False Positive (FP)
    rejected_defaulted = (y_pred == 1) & (y_true == 1)  # True Positive (TP)

    # Cash flows
    profit_earned   = np.sum(expected_profit[approved_paid])
    losses_incurred = np.sum(real_lgd[approved_defaulted])
    profit_missed   = np.sum(expected_profit[rejected_paid])
    losses_avoided  = np.sum(real_lgd[rejected_defaulted])
    net_profit      = profit_earned - losses_incurred

    print(f"\n{'='*65}")
    print(f"FINANCIAL IMPACT & BANKING P&L REPORT — {model_name.upper()}")
    print(f"{'='*65}")
    print(f"  (+) Expected Net Profit Earned (TN): €{profit_earned:>14,.2f}")
    print(f"  (-) Realized Losses LGD (FN):        €{losses_incurred:>14,.2f}")
    print(f"  (-) Opportunity Cost / Missed (FP):  €{profit_missed:>14,.2f}")
    print(f"  (+) Default Losses Avoided (TP):     €{losses_avoided:>14,.2f}")
    print(f"  {'-'*55}")
    print(f"  TOTAL NET PROFIT:                    €{net_profit:>14,.2f}")
    print(f"{'='*65}\n")

    return {
        "profit_earned": profit_earned,
        "losses_incurred": losses_incurred,
        "profit_missed": profit_missed,
        "losses_avoided": losses_avoided,
        "net_profit": net_profit,
        "counts": {
            "TN": int(np.sum(approved_paid)),
            "FN": int(np.sum(approved_defaulted)),
            "FP": int(np.sum(rejected_paid)),
            "TP": int(np.sum(rejected_defaulted)),
        }
    }
