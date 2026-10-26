"""Loan-selection strategy: invest only in loans whose predicted NAR >= threshold M (paper, Sec. VIII)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import MIN_COVERAGE, RESULTS_DIR, TARGET_RETURN


def investment_curve(pred, actual) -> pd.DataFrame:
    """Rank loans by predicted NAR (high -> low); return avg ACTUAL NAR as we invest in more loans."""
    order = np.argsort(-np.asarray(pred))
    a = np.asarray(actual)[order]
    n = np.arange(1, len(a) + 1)
    return pd.DataFrame({"pct_invested": n / len(a), "avg_return": np.cumsum(a) / n,
                         "threshold": np.asarray(pred)[order]})


def choose_threshold(val_pred, val_actual, target=TARGET_RETURN, min_cov=MIN_COVERAGE) -> float:
    """Pick the lowest threshold M (= most loans invested) that still achieves `target` avg return on VALIDATION."""
    c = investment_curve(val_pred, val_actual)
    ok = c[(c.avg_return >= target) & (c.pct_invested >= min_cov)]
    if ok.empty:   # target unreachable -> fall back to the best-return point with enough coverage
        ok = c[c.pct_invested >= min_cov]
        ok = ok.loc[[ok.avg_return.idxmax()]]
    return float(ok.iloc[-1].threshold)


def apply_threshold(pred, actual, M: float) -> dict:
    pred, actual = np.asarray(pred), np.asarray(actual)
    sel = pred >= M
    return {"threshold_M": M, "pct_loans_invested": float(sel.mean()), "n_invested": int(sel.sum()),
            "avg_return_selected": float(actual[sel].mean()) if sel.any() else float("nan"),
            "avg_return_all_loans": float(actual.mean())}


def plot_curves(curves: dict, M: float | None = None, fname="strategy_return_curve.png"):
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(7, 4.5))
    for name, c in curves.items():
        plt.plot(c.pct_invested * 100, c.avg_return * 100, label=name)
    plt.axhline(TARGET_RETURN * 100, ls="--", c="grey", label=f"{TARGET_RETURN:.0%} benchmark")
    plt.xlabel("% of loans invested (highest predicted NAR first)"); plt.ylabel("annualised return (%)")
    plt.title("Investment return vs. percentage of loans invested"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / fname, dpi=150); plt.close()
