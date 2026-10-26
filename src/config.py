"""Central configuration shared by all scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.15      # final hold-out
VAL_SIZE = 0.15       # used for hyper-parameter / threshold selection (paper's "future work")

# Raw LendingClub columns we read (same names as the provided dataset.xlsx 'accepted' sheet).
RAW_COLUMNS = [
    "loan_amnt", "term", "int_rate", "grade", "emp_length", "annual_inc",
    "issue_d", "loan_status", "dti", "fico_range_low", "fico_range_high",
    "total_pymnt",
    "last_pymnt_d",  # optional: only in the full LendingClub CSV; used for exact NAR
]

# Model inputs after feature engineering (see preprocessing.py).
# NOTE: total_pymnt / last_pymnt_d / loan_status are NEVER features (they leak the outcome).
NUMERIC_FEATURES = [
    "loan_amnt", "int_rate", "annual_inc", "dti", "fico_avg",
    "term_months", "emp_length_years", "issue_days", "loan_to_income",
]
CATEGORICAL_FEATURES = ["grade"]

# Loan selection strategy
TARGET_RETURN = 0.10   # S&P-500 style benchmark used in the reference paper
MIN_COVERAGE = 0.01    # never pick a threshold that invests in <1% of loans
