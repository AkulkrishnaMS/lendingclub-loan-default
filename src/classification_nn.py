"""Classification model 2: Neural Network (MLP) -> predicts Fully Paid vs Default.

Config A reproduces the paper: 4 hidden layers (10,10,5,3), sigmoid ('logistic'), Adam, batch 200,
lr 0.001, L2 alpha 1e-4.  The paper's NN overfit, so we add early stopping and also try
a few alternative architectures, choosing the best on the validation set.
"""
import joblib
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .cli import get_args
from .config import MODELS_DIR, RANDOM_STATE
from .data_loader import build_dataset, load_raw
from .evaluation import classification_metrics
from .preprocessing import build_preprocessor, split_data

TAG = "nn"
CONFIGS = {
    "paper (10,10,5,3) sigmoid": dict(hidden_layer_sizes=(10, 10, 5, 3), activation="logistic"),
    "(64,32) relu":              dict(hidden_layer_sizes=(64, 32), activation="relu"),
    "(32,16,8) relu":            dict(hidden_layer_sizes=(32, 16, 8), activation="relu"),
}


def run(data=None, years=None):
    df = build_dataset(load_raw(data), years)
    s = split_data(df)
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _) = s["train"], s["val"], s["test"]

    best, best_f1, best_name = None, -1, None
    for name, cfg in CONFIGS.items():
        m = Pipeline([("prep", build_preprocessor()),
                      ("clf", MLPClassifier(solver="adam", batch_size=200, learning_rate_init=0.001,
                                            alpha=1e-4, max_iter=200, early_stopping=True,
                                            n_iter_no_change=10, random_state=RANDOM_STATE, **cfg))])
        m.fit(Xtr, ytr)
        f1 = f1_score(yva, m.predict(Xva), average="macro")
        print(f"[nn] {name:<28} val macro-F1={f1:.4f}")
        if f1 > best_f1:
            best, best_f1, best_name = m, f1, name

    print(f"[nn] selected: {best_name}")
    metrics = classification_metrics(best, {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}, TAG)
    metrics["selected_config"] = best_name
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(best, MODELS_DIR / f"{TAG}.joblib")
    return metrics


if __name__ == "__main__":
    a = get_args("Train Neural Network classifier")
    run(a.data, tuple(a.years) if a.years else None)
