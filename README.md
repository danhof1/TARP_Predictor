# Market Classification: Structural Trading Changes During TARP

## Overview

This project analyzes changes in market behavior before and after the implementation of the Troubled Asset Relief Program (TARP) on October 3, 2008. It uses high-frequency trading data and machine learning techniques to classify each day as either pre- or post-TARP. The project consists of three core components:

1. **Market Analysis** (`Market_TARP_Volatility.ipynb`)
2. **Hyperparameter Optimization** (`Optuna_Assignment1TaleofTwoBozos.ipynb`)
3. **Model Training and Results** (`Assignment1TaleofTwoBozosMaster.ipynb`)

---

## 1. Market Analysis

### Objective
To explore market microstructure features and visualize regime changes around the TARP intervention.

### Dataset
- 10-minute cumulative returns of ETFs: SPY, QQQ, XLF, XLE, XLY
- Period: Oct 9, 2007 – Mar 31, 2009
- Label: `post_TARP` (0 = before, 1 = after)

### Features Extracted
- **Intraday Return Volatility** for each ETF and aggregate
- **Cumulative Absolute Returns**
- **Busiest Volume Spike Time** and **Magnitude** per ETF
- **Statistical Measures**: mean, std, skewness, kurtosis

### Key Findings
- **Volatility dropped** significantly after TARP.
- **Return magnitudes** were larger pre-TARP, indicating instability.
- Intraday trading patterns shifted across sectors, especially financials (XLF).
- Polynomial regression trends supported a regime change on October 3, 2008.

---

## 2. Hyperparameter Optimization (Optuna)

### Objective
Tune hyperparameters for classifiers using `Optuna` to improve classification of pre/post-TARP days.

### Methods
- Search space included:
  - Learning rate
  - Dropout rate
  - Regularization parameters (`alpha`, `lambda`)
  - Tree-specific parameters (`max_depth`, `gamma`, etc.)
- 5-fold cross-validation on training data
- Objective: Maximize AUC-ROC

### Outcome
- Optuna generated optimal values, passed to `GridSearchCV` for fine-tuning.
- Best configuration used for the final model training.

---

## 3. Final Model & Results

### Model Used
- **XGBoost Classifier** (`xgb.XGBClassifier(eval_metric='auc', random_state=42)`)

### Best Hyperparameters (via Optuna + GridSearchCV)
```python
{
  'max_depth': 5,
  'learning_rate': 0.128,
  'subsample': 0.948,
  'colsample_bytree': 0.953,
  'min_child_weight': 3,
  'gamma': 1.423,
  'alpha': 0.071,    # L1 regularization
  'lambda': 1.251    # L2 regularization
}
```

### Threshold Adjustment
- Prediction threshold set to **0.3** to improve sensitivity to post-TARP days.

### Evaluation
- Model performance evaluated with AUC-ROC and classification report.
- ROC curve was saved and analyzed.

---

## Conclusion

- A combination of financial domain knowledge and machine learning was used to detect regime shifts in trading behavior.
- Feature engineering around volatility, returns, and volume timing captured the TARP impact.
- Optimization via Optuna significantly boosted model performance.
- The final XGBoost classifier successfully dated market days with high accuracy, supporting the hypothesis of a structural market change post-TARP.

---

## Future Work

- Expand feature set to include macroeconomic indicators or order book data.
- Apply causal inference techniques to isolate TARP’s true market impact.
- Incorporate time-aware models (e.g., LSTMs or transformers) for sequential pattern learning.
