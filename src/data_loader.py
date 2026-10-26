"""Load raw LendingClub data (full CSV, the provided .xlsx sample, or synthetic) and build labels."""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, RAW_COLUMNS

CANDIDATES = ["accepted_2007_to_2018Q4.csv", "accepted.csv", "synthetic_loans.csv", "dataset_sample.xlsx"]


def find_data_file(path: str | None = None) -> Path:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for name in CANDIDATES:
        p = DATA_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "No dataset found in data/. Download LendingClub 'accepted' CSV into data/ "
        "or run: python generate_synthetic_data.py"
    )


def _read_xlsx(path: Path) -> pd.DataFrame:
    """The provided sample has a title row above the header; find the header row automatically."""
    xl = pd.ExcelFile(path)
    sheet = "accepted" if "accepted" in xl.sheet_names else xl.sheet_names[0]
    raw = xl.parse(sheet, header=None)
    hdr = next(i for i in range(len(raw)) if "loan_amnt" in raw.iloc[i].astype(str).values)
    df = raw.iloc[hdr + 1:].copy()
    df.columns = raw.iloc[hdr].astype(str).values
    return df.reset_index(drop=True)


def load_raw(path: str | None = None) -> pd.DataFrame:
    p = find_data_file(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = _read_xlsx(p)
        df = df[[c for c in RAW_COLUMNS if c in df.columns]]
    else:
        df = pd.read_csv(p, usecols=lambda c: c in RAW_COLUMNS, low_memory=False)
    print(f"[data] loaded {len(df):,} rows from {p.name}")
    return df


def _parse_issue_date(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, format="%b-%Y", errors="coerce")       # 'Dec-2015' (full CSV)
    return out.fillna(pd.to_datetime(s, format="mixed", errors="coerce"))           # ISO / Excel datetimes


def build_dataset(df: pd.DataFrame, years: tuple[int, int] | None = None) -> pd.DataFrame:
    """Keep finalised loans only, create the classification label and the regression label (NAR)."""
    df = df.copy()
    status = df["loan_status"].astype(str)
    paid = status.str.contains("Fully Paid", case=False)
    bad = status.str.contains("Charged Off|Default", case=False)
    df = df[paid | bad].copy()                       # drop Current / Late / In Grace Period ...
    df["label"] = paid[df.index].astype(int)         # 1 = Fully Paid, 0 = Default / Charged Off

    df["issue_d"] = _parse_issue_date(df["issue_d"])
    if years is not None:
        df = df[df["issue_d"].dt.year.between(*years)]

    for c in ["loan_amnt", "annual_inc", "dti", "fico_range_low", "fico_range_high", "total_pymnt"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["int_rate"] = pd.to_numeric(df["int_rate"].astype(str).str.replace("%", ""), errors="coerce")
    df = df.dropna(subset=["loan_amnt", "total_pymnt", "issue_d"])
    df = df[df["loan_amnt"] > 0].reset_index(drop=True)

    df["nar"] = compute_nar(df)
    print(f"[data] finalised loans: {len(df):,} | Fully Paid: {df.label.mean():.1%} | Default: {1 - df.label.mean():.1%}")
    return df


def compute_nar(df: pd.DataFrame) -> pd.Series:
    """Net annualised return  y = (total_payment / loan_amount) ** (365 / D) - 1   (paper, Sec. VI).

    D = days between issue and last payment when `last_pymnt_d` exists (full dataset);
    otherwise D falls back to the full loan term (documented approximation).
    """
    term_days = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce") * 30.4375
    D = term_days
    if "last_pymnt_d" in df.columns:
        last = _parse_issue_date(df["last_pymnt_d"])
        exact = (last - df["issue_d"]).dt.days
        D = exact.where(exact > 0, term_days)
    D = D.fillna(36 * 30.4375).clip(lower=30)
    nar = (df["total_pymnt"] / df["loan_amnt"]).clip(lower=1e-6) ** (365.0 / D) - 1
    return nar.clip(-1, 1)   # clip extreme outliers from very short loans
