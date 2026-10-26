"""Run the whole pipeline in order:  logreg -> neural net -> random-forest regression + strategy -> summary table.

    python run_all.py                          # auto-detects data in data/
    python run_all.py --data data/accepted_2007_to_2018Q4.csv --years 2012 2015
"""
import pandas as pd

from src import classification_logreg, classification_nn, regression_rf
from src.cli import get_args
from src.config import RESULTS_DIR


def main():
    a = get_args("Run all models")
    yrs = tuple(a.years) if a.years else None
    lr = classification_logreg.run(a.data, yrs)
    nn = classification_nn.run(a.data, yrs)
    rf = regression_rf.run(a.data, yrs)

    rows = []
    for name, m in [("Logistic Regression", lr), ("Neural Network", nn)]:
        t = m["test"]
        rows.append({"model": name, "task": "classification", "precision(w)": t["weighted_avg"]["precision"],
                     "recall(w)": t["weighted_avg"]["recall"], "f1(w)": t["weighted_avg"]["f1-score"],
                     "default recall": t["default"]["recall"], "AUC": t["roc_auc"]})
    rows.append({"model": "Random Forest", "task": "regression", "test MSE": rf["test"]["mse"],
                 "test R2": rf["test"]["r2"], "strategy return": rf["strategy"]["test"]["avg_return_selected"],
                 "% loans invested": rf["strategy"]["test"]["pct_loans_invested"]})
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print("\n================ SUMMARY (test set) ================")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
