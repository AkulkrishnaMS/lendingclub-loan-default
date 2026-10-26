"""Smoke tests: run with  pytest -q"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.data_loader import build_dataset, load_raw, compute_nar
from src.preprocessing import make_features, build_preprocessor
from src.config import DATA_DIR


def test_excel_sample_loads_and_labels():
    df = build_dataset(load_raw(str(DATA_DIR / "dataset_sample.xlsx")))
    assert len(df) == 5 and set(df.label) == {0, 1}
    assert df.nar.between(-1, 1).all()


def test_no_leakage_columns_in_features():
    df = build_dataset(load_raw(str(DATA_DIR / "dataset_sample.xlsx")))
    X = make_features(df)
    assert not {"total_pymnt", "loan_status", "label", "nar"} & set(X.columns)


def test_preprocessor_outputs_finite():
    df = build_dataset(load_raw(str(DATA_DIR / "dataset_sample.xlsx")))
    Z = build_preprocessor().fit_transform(make_features(df))
    assert np.isfinite(Z.toarray() if hasattr(Z, "toarray") else Z).all()
