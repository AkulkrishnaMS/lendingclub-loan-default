# Write-up outline (max 2 pages, submit as PDF)  - fill numbers from results/summary.csv

**1. Problem statement** - Predict P2P loan default (classification: LogReg, NN) and loan profitability (regression: Random Forest), then use predicted return to select loans. Motivation: investors must assess credit risk across thousands of loans.

**2. Dataset** - LendingClub accepted loans, issued 2012-2015, finalised only. N = __, % default = __. Features: loan_amnt, term, int_rate, grade, emp_length, annual_inc, dti, FICO avg, issue date, loan-to-income. Split 70/15/15.

**3. Approach** - preprocessing; labels (Fully Paid=1, Default/Charged Off=0; NAR formula); model choices and why; hyper-parameter tuning on validation; threshold strategy.

**4. Implementation overview** - repo structure, pipeline order, tech (scikit-learn, pandas), no-leakage design.

**5. Results** - table: LogReg vs NN (precision/recall/F1 weighted, default-class recall, AUC); RF (MSE, R2, depth); strategy (M, % invested, return vs. all-loans). Include 1 figure (strategy curve or confusion matrices).

**6. Conclusions** - which classifier is better and why; RF R2 vs paper (0.315); strategy return vs 10% benchmark; limitations (fewer features, random split not time-based, class imbalance); future work (time-based test on 2016-18, NLP on descriptions, macro features).

---
# Presentation outline (~8 slides)
1 Title/team | 2 Problem & motivation | 3 Data & EDA | 4 Preprocessing | 5 Models (LogReg, NN, RF) | 6 Results table + plots | 7 Strategy result | 8 Conclusions & contributions. Prepare a live demo: `python run_all.py` (or load `models/*.joblib`).
