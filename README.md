

# RiskShield — AI Fraud Risk Manager

An end-to-end fraud detection system for mobile money transactions. RiskShield takes 6.3 million raw transactions, engineers behavioural features, trains a gradient boosting classifier with a time-based split, and wraps the model in a decision engine that turns probabilities into operational actions (`APPROVE` / `BLOCK / INVESTIGATE`).

The project is benchmarked against the dataset's existing rule-based flagging system, which catches almost nothing.

---

## Results

Final evaluation on a held-out test set of 103,573 transactions (time steps > 600, never seen during training or threshold selection):

| Metric | RiskShield | Existing rule-based system |
|---|---|---|
| Precision | 0.9994 | 1.0000 |
| Recall | 0.9994 | 0.0019 |
| F1 score | 0.9994 | 0.0039 |
| Accuracy | 1.0000 | — |

Confusion matrix at a 0.70 risk threshold:

| | Predicted legitimate | Predicted fraud |
|---|---|---|
| **Actual legitimate** | 101,972 | 1 |
| **Actual fraud** | 1 | 1,599 |

### Business impact

| | Amount |
|---|---|
| Actual fraud value | ₹2,658,440,348.08 |
| Fraud value detected | ₹2,658,041,303.00 |
| Fraud value missed | ₹399,045.08 |
| False positives | 1 transaction |

**Fraud amount capture rate: 99.98%**

The legacy rule-based flag caught 0.19% of fraud cases. RiskShield recovers essentially all of the fraud value while blocking a single legitimate transaction across the entire test period.

---

## Dataset

PaySim synthetic mobile money dataset (`Fraud.csv`), 6,362,620 rows × 11 columns.

- Fraud cases: 8,213 (0.13% of all transactions)
- Transaction types: `PAYMENT`, `TRANSFER`, `CASH_OUT`, `CASH_IN`, `DEBIT`
- Fraud only ever occurs in `TRANSFER` and `CASH_OUT`
- `step` represents one hour of simulated time

The file is not included in this repo. Download it from Kaggle ([PaySim synthetic financial datasets](https://www.kaggle.com/datasets/ealaxi/paysim1)) and place `Fraud.csv` in the project root.

---

## Approach

### 1. Exploratory analysis
Class balance, fraud rate by transaction type, amount distributions, fraud over time, and an audit of the `isFlaggedFraud` column. Balance-error columns (`oldbalance - amount - newbalance`) are computed to check ledger consistency and expose the drain-the-account pattern behind most fraud.

### 2. Feature engineering
Only pre-transaction information is used, so nothing leaks from the outcome of the transfer:

| Feature | Meaning |
|---|---|
| `amount_log` | log1p of amount, to compress the heavy tail |
| `sender_remaining_balance` | `oldbalanceOrg - amount` |
| `amount_to_sender_balance` | share of the sender's balance being moved (clipped at 10) |
| `amount_to_destination_balance` | amount relative to recipient's balance (clipped at 10) |
| `origin_balance_zero`, `destination_balance_zero` | empty-account flags |
| `is_large_transaction` | amount > 100,000 |
| `is_transfer_or_cashout` | the only two types where fraud occurs |
| `type_*` | one-hot encoded transaction type |

Post-transaction balances (`newbalanceOrig`, `newbalanceDest`) are deliberately excluded — they would not be known at scoring time.

### 3. Time-based split
No random shuffling. The split respects chronology so the model is always predicting forward in time:

| Split | Time steps | Rows | Fraud | Fraud rate |
|---|---|---|---|---|
| Train | ≤ 500 | 6,061,807 | 5,561 | 0.09% |
| Validation | 501–600 | 197,240 | 1,052 | 0.53% |
| Test | > 600 | 103,573 | 1,600 | 1.54% |

### 4. Class imbalance
The training set is downsampled to all fraud cases plus 300,000 randomly sampled legitimate transactions, and models are trained with `class_weight="balanced"`.

### 5. Modelling
- **Baseline:** Logistic Regression with standardised features
- **Final:** `HistGradientBoostingClassifier` (200 iterations, learning rate 0.08, 31 leaf nodes, L2 = 1.0)

### 6. Threshold selection
Thresholds from 0.10 to 0.90 are swept on the validation set, scored two ways: standard precision/recall/F1, and a cost model that prices a missed fraud at its full transaction value and a false positive at 1% of transaction value. `0.70` is selected as the operating threshold and then locked before touching the test set.

### 7. Decision engine
The model output becomes an actionable decision rather than a raw probability:

| Risk score | Level | Decision |
|---|---|---|
| < 0.20 | LOW | APPROVE |
| 0.20 – 0.50 | MEDIUM | APPROVE |
| 0.50 – 0.70 | HIGH | APPROVE |
| ≥ 0.70 | CRITICAL | BLOCK / INVESTIGATE |

---

## Usage

Score a single transaction:

```python
result = assess_transaction(final_model, X_test.iloc[[0]])
# {'risk_score': 0.0002, 'risk_level': 'LOW', 'decision': 'APPROVE'}
```

Score a batch:

```python
val_results = score_transactions(final_model, X_val)
# returns a DataFrame with risk_score, risk_level, decision
```

---

## Running the notebook

```bash
git clone https://github.com/AishaaFatima/RiskShield.git
cd RiskShield
pip install -r requirements.txt
jupyter notebook RiskShield.ipynb
```


### Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## Project structure

```
RiskShield/
├── RiskShield.ipynb    # full analysis, modelling and evaluation
├── Fraud.csv           # dataset (not tracked — download separately)
├── requirements.txt
└── README.md
```

---

## Limitations

A few things worth being upfront about:

- **The dataset is synthetic.** PaySim fraud follows a very regular pattern — the account is drained completely, so `amount` almost exactly equals `oldbalanceOrg`. The engineered ratio features pick this up cleanly, which is a large part of why the scores are near-perfect. Real-world fraud is messier and these numbers should not be read as an expected production result.
- **No account-level history.** Features are computed per transaction. A production system would add velocity features, recipient reputation, device and location signals, and per-customer baselines.
- **The threshold is cost-model dependent.** Pricing false positives at 1% of transaction value is an assumption, not a measured cost. A different investigation cost would shift the optimal threshold.
- **Concept drift is not handled.** The model is trained once and evaluated forward. Fraud patterns change; a live deployment would need scheduled retraining and monitoring for score distribution drift.

---

