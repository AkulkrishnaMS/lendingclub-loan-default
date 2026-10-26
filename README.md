# LendingClub Loan Default & Profitability Prediction  (UE24CS352A - ML Mini-Project)

Re-implementation / extension of *"LendingClub Loan Default and Profitability Prediction"* (Li & Han).

| Task | Model(s) | Target |
|---|---|---|
| Classification | **Logistic Regression** and **Neural Network (MLP)** - trained separately | Fully Paid (1) vs Default / Charged Off (0) |
| Regression | **Random Forest** | Net Annualised Return (NAR) = (total_pymnt / loan_amnt)^(365/D) - 1 |
| Strategy | Invest only in loans with predicted NAR >= M | Avg. realised return on held-out loans |

## Repository layout
```
run_all.py                  # runs everything in order and prints a summary table
generate_synthetic_data.py  # fake data for testing the pipeline
src/
  config.py                 # paths, seed, split sizes, feature lists
  data_loader.py            # read CSV/XLSX, keep finalised loans, labels, NAR
  preprocessing.py          # feature engineering, imputation, scaling, one-hot, train/val/test split
  evaluation.py             # metrics, confusion matrix, ROC
  classification_logreg.py  # model 1
  classification_nn.py      # model 2
  regression_rf.py          # model 3 (+ calls strategy.py)
  strategy.py               # loan selection threshold M + return curve
tests/test_pipeline.py      # smoke tests
data/  results/  models/  docs/
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run (in this order)
```bash
# 0. (optional) sanity-check the code
pytest -q

# 1. Put the real LendingClub CSV in data/  (see data/README.md). No data yet? -> python generate_synthetic_data.py

# 2. Everything at once
python run_all.py --data data/accepted_2007_to_2018Q4.csv --years 2012 2015

# ... or each model separately
python -m src.classification_logreg --data data/accepted_2007_to_2018Q4.csv --years 2012 2015
python -m src.classification_nn      --data data/accepted_2007_to_2018Q4.csv --years 2012 2015
python -m src.regression_rf          --data data/accepted_2007_to_2018Q4.csv --years 2012 2015
```
Outputs: `results/` (metrics JSON, confusion matrices, ROC curves, RF feature importance, strategy curve, `summary.csv`) and `models/*.joblib`.

## Method highlights
- **Finalised loans only** (Fully Paid / Charged Off / Default); "Current", "Late" etc. dropped (as in the paper).
- **No leakage**: `total_pymnt`, `last_pymnt_d`, `loan_status` are used only to build labels, never as features.
- **Pipeline fitted on train only**: median imputation (+missing indicators), standardisation, one-hot grade.
- **Train / validation / test = 70 / 15 / 15** (stratified). The paper had no dev set (its own "future work"); we tune hyper-parameters (C, MLP architecture, RF depth) and the strategy threshold M on validation, and report once on test.
- **Logistic Regression**: L2, `class_weight="balanced"`, C tuned.
- **Neural Network**: paper config (10,10,5,3 sigmoid) vs. two alternatives, with early stopping to reduce the overfitting the paper observed.
- **Random Forest**: 200 trees, depth tuned over {4, 8, 10, 12}; metrics MSE / RMSE / R2.
- **Strategy**: M is the lowest threshold whose *validation* return >= 10% (S&P 500 benchmark); evaluated on test.

## Known simplifications (state these in the write-up)
- Only the columns in the provided sample schema are used (the paper used ~1,100 features after one-hot encoding).
- Imputation uses median + indicator flags rather than the paper's mean/zero/max-set scheme.
- NAR uses `last_pymnt_d` when present; otherwise D = full loan term.
- NAR is clipped to [-1, 1] to limit outliers from very short loans.
