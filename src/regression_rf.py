"""Regression model: Random Forest predicting net annualised return (NAR), then loan-selection strategy."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from . import strategy
from .cli import get_args
from .config import MODELS_DIR, RANDOM_STATE, RESULTS_DIR
from .data_loader import build_dataset, load_raw
from .evaluation import regression_metrics, save_json
from .preprocessing import build_preprocessor, split_data

TAG = "rf_regressor"
MAX_DEPTHS = [4, 8, 10, 12]   # paper tried 4, 8, 10


def run(data=None, years=None):
    df = build_dataset(load_raw(data), years)
    s = split_data(df)
    (Xtr, _, rtr), (Xva, _, rva), (Xte, _, rte) = s["train"], s["val"], s["test"]

    results, best, best_mse = [], None, np.inf
    for d in MAX_DEPTHS:
        m = Pipeline([("prep", build_preprocessor()),
                      ("rf", RandomForestRegressor(n_estimators=200, max_depth=d, min_samples_leaf=5,
                                                   max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE))])
        m.fit(Xtr, rtr)
        tr, va = regression_metrics(rtr, m.predict(Xtr)), regression_metrics(rva, m.predict(Xva))
        results.append({"max_depth": d, "train_mse": tr["mse"], "val_mse": va["mse"],
                        "train_r2": tr["r2"], "val_r2": va["r2"]})
        print(f"[rf] depth={d:<3} train MSE={tr['mse']:.4f} R2={tr['r2']:.3f} | val MSE={va['mse']:.4f} R2={va['r2']:.3f}")
        if va["mse"] < best_mse:
            best, best_mse = m, va["mse"]

    depth_table = pd.DataFrame(results)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    depth_table.to_csv(RESULTS_DIR / "rf_depth_search.csv", index=False)
    print(f"[rf] best max_depth = {best.named_steps['rf'].max_depth}")

    metrics = {k: regression_metrics(y, best.predict(X)) for k, (X, y) in
               {"train": (Xtr, rtr), "val": (Xva, rva), "test": (Xte, rte)}.items()}
    metrics["best_max_depth"] = best.named_steps["rf"].max_depth
    print("\n=== RF regressor ===")
    for k in ("train", "val", "test"):
        print(f"{k:5s} MSE={metrics[k]['mse']:.4f} RMSE={metrics[k]['rmse']:.3f} R2={metrics[k]['r2']:.3f}")

    # ---- feature importance
    names = best.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(best.named_steps["rf"].feature_importances_, index=names).sort_values().tail(12)
    imp.plot.barh(figsize=(6, 4.5), title="RF feature importance (top 12)")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "rf_feature_importance.png", dpi=150); plt.close()

    # ---- loan selection strategy: choose M on VALIDATION, report on TEST
    pva, pte, ptr = best.predict(Xva), best.predict(Xte), best.predict(Xtr)
    M = strategy.choose_threshold(pva, rva.values)
    metrics["strategy"] = {"val": strategy.apply_threshold(pva, rva.values, M),
                           "test": strategy.apply_threshold(pte, rte.values, M)}
    strategy.plot_curves({"train": strategy.investment_curve(ptr, rtr.values),
                          "test": strategy.investment_curve(pte, rte.values)}, M)
    print("\n=== Loan selection strategy (M chosen on validation) ===")
    for k, v in metrics["strategy"].items():
        print(f"{k:5s} M={v['threshold_M']:.3f} | invests in {v['pct_loans_invested']:.2%} of loans | "
              f"avg return selected={v['avg_return_selected']:.2%} vs all loans={v['avg_return_all_loans']:.2%}")

    save_json(metrics, f"{TAG}_metrics.json")
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(best, MODELS_DIR / f"{TAG}.joblib")
    return metrics


if __name__ == "__main__":
    a = get_args("Train Random Forest NAR regressor + loan selection strategy")
    run(a.data, tuple(a.years) if a.years else None)
