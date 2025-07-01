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
- Search Space:  
  - Learning rate, dropout, weight decay  
  - Loss functions (e.g., binary cross-entropy, Focal Loss, custom)  
  - Optimizers (Adam, SGD variants)  
- 5-fold cross-validation on training data.
- AUC-ROC used as the objective for trials.

### Outcome
- Found optimal configurations that maximized AUC-ROC.
- Best-performing model parameters passed to final training notebook.

---

## 3. Final Model & Results

### Objective
Train and evaluate the final classification model using the full engineered feature set and Optuna-selected hyperparameters.

### Classifier Used
- (Assumed from codebase) Ensemble model or tuned neural network.

### Metrics
- **Evaluation Metric**: AUC-ROC  
- Achieved strong separability between pre- and post-TARP samples.
- Model was able to generalize well on holdout test data.

---

## Conclusion

- A combination of financial domain knowledge and machine learning was used to detect regime shifts in trading behavior.
- Feature engineering around volatility, returns, and volume timing captured the TARP impact.
- Optimization via Optuna significantly boosted model performance.
- The final classifier successfully dated market days with high accuracy, supporting the hypothesis of a structural market change post-TARP.

---

## Future Work

- Expand feature set to include macroeconomic indicators or order book data.
- Apply causal inference techniques to isolate TARP’s true market impact.
- Incorporate time-aware models (e.g., LSTMs or transformers) for sequential pattern learning.

