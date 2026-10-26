"""Classification model 1: Logistic Regression (L2, balanced class weights)  -> predicts Fully Paid vs Default."""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from .cli import get_args
from .config import MODELS_DIR, RANDOM_STATE
from .data_loader import build_dataset, load_raw
from .evaluation import classification_metrics
from .preprocessing import build_preprocessor, split_data

TAG = "logreg"


def run(data=None, years=None):
    df = build_dataset(load_raw(data), years)
    s = split_data(df)
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _) = s["train"], s["val"], s["test"]

    # Tune inverse regularisation strength C on the validation set (macro-F1 so the minority class counts).
    best, best_f1 = None, -1
    for C in [0.01, 0.1, 1, 10]:
        m = Pipeline([("prep", build_preprocessor()),
                      ("clf", LogisticRegression(C=C, penalty="l2", class_weight="balanced",
                                                 max_iter=1000, random_state=RANDOM_STATE))])
        m.fit(Xtr, ytr)
        f1 = f1_score(yva, m.predict(Xva), average="macro")
        print(f"[logreg] C={C:<5} val macro-F1={f1:.4f}")
        if f1 > best_f1:
            best, best_f1 = m, f1

    print(f"[logreg] best C = {best.named_steps['clf'].C}")
    metrics = classification_metrics(best, {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}, TAG)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(best, MODELS_DIR / f"{TAG}.joblib")
    return metrics


if __name__ == "__main__":
    a = get_args("Train Logistic Regression classifier")
    run(a.data, tuple(a.years) if a.years else None)
