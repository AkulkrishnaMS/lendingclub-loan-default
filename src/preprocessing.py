"""Feature engineering + sklearn preprocessing pipeline and the shared train/val/test split."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE, TEST_SIZE, VAL_SIZE

EPOCH = pd.Timestamp("1970-01-01")


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["loan_amnt"] = df["loan_amnt"]
    X["int_rate"] = df["int_rate"]
    X["annual_inc"] = df["annual_inc"]
    X["dti"] = df["dti"]
    X["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2   # low/high are ~collinear (paper Fig.1)
    X["term_months"] = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    emp = df["emp_length"].astype(str).str.lower().str.replace("< 1 year", "0", regex=False)
    X["emp_length_years"] = pd.to_numeric(emp.str.extract(r"(\d+)")[0], errors="coerce")
    X["issue_days"] = (df["issue_d"] - EPOCH).dt.days                    # dates -> days since epoch (paper)
    X["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
    X["grade"] = df["grade"].astype(str)
    return X[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Fitted on TRAIN only (inside each model Pipeline) so no information leaks from val/test."""
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),  # missing-value flag kept as a feature
        ("scale", StandardScaler()),                                       # zero mean / unit variance (paper)
    ])
    cat = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("num", num, NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)])


def split_data(df: pd.DataFrame):
    """Stratified train / val / test. Identical for every script (same seed) so models are comparable."""
    X, y_cls, y_reg = make_features(df), df["label"], df["nar"]
    idx = np.arange(len(df))
    tr_val, te = train_test_split(idx, test_size=TEST_SIZE, stratify=y_cls, random_state=RANDOM_STATE)
    tr, va = train_test_split(tr_val, test_size=VAL_SIZE / (1 - TEST_SIZE),
                              stratify=y_cls.iloc[tr_val], random_state=RANDOM_STATE)
    out = {}
    for name, i in [("train", tr), ("val", va), ("test", te)]:
        out[name] = (X.iloc[i], y_cls.iloc[i], y_reg.iloc[i])
    print("[split]", {k: len(v[0]) for k, v in out.items()})
    return out
