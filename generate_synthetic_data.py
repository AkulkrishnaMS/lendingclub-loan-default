"""Create a SYNTHETIC LendingClub-like CSV so the pipeline can be tested without the 1.6 GB real file.
Same columns as the provided dataset.xlsx. Results on this data are NOT meaningful for the report.

    python generate_synthetic_data.py --n 30000
"""
import argparse
import numpy as np
import pandas as pd
from src.config import DATA_DIR


def main(n=30000, seed=0):
    r = np.random.default_rng(seed)
    fico = np.clip(r.normal(695, 30, n), 620, 840).round(-0).astype(int)
    grade_idx = np.clip(((760 - fico) / 22).astype(int) + r.integers(-1, 2, n), 0, 6)
    grade = np.array(list("ABCDEFG"))[grade_idx]
    int_rate = (6 + 2.6 * grade_idx + r.normal(0, 0.8, n)).round(2)
    term = r.choice([36, 60], n, p=[0.75, 0.25])
    loan = (r.integers(1, 36, n) * 1000).astype(float)
    income = np.exp(r.normal(11.1, 0.5, n)).round()
    dti = np.clip(r.normal(17, 8, n), 0, 40).round(2)
    emp = r.choice(["< 1 year", "1 year", "2 years", "3 years", "5 years", "8 years", "10+ years"], n)
    issue = pd.to_datetime("2012-01-01") + pd.to_timedelta(r.integers(0, 4 * 365, n), unit="D")

    z = (-1.9 + 0.09 * (int_rate - 12) + 0.03 * (dti - 17) - 0.012 * (fico - 695)
         + 0.8 * (loan / income) + 0.35 * (term == 60) + r.normal(0, 0.6, n))
    p_def = 1 / (1 + np.exp(-z))
    default = r.random(n) < p_def
    interest_total = loan * int_rate / 100 * term / 12 * 0.55
    total_pymnt = np.where(default, loan * r.uniform(0.1, 0.75, n), loan + interest_total).round(2)

    status = np.where(default, np.where(r.random(n) < .9, "Charged Off", "Default"), "Fully Paid")
    status = np.where(r.random(n) < 0.12, "Current", status)  # unfinished loans -> should be filtered out
    df = pd.DataFrame({
        "loan_amnt": loan, "term": [f" {t} months" for t in term], "int_rate": int_rate, "grade": grade,
        "emp_length": emp, "annual_inc": income, "issue_d": issue.strftime("%b-%Y"), "loan_status": status,
        "dti": dti, "fico_range_low": fico, "fico_range_high": fico + 4, "total_pymnt": total_pymnt})
    df.loc[r.random(n) < 0.05, "emp_length"] = np.nan
    df.loc[r.random(n) < 0.01, "dti"] = np.nan
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "synthetic_loans.csv", index=False)
    print(f"wrote {len(df):,} rows -> data/synthetic_loans.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30000)
    main(ap.parse_args().n)
