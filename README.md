# 📈 Tale of Two Markets: Market Volatility and Structural Change During TARP

This repository contains a suite of notebooks focused on analyzing and modeling structural changes in trading behavior around the 2008 financial crisis — specifically in response to the Troubled Asset Relief Program (TARP).

---

## 🧠 Project Objective

The core task is to build a **binary classification model** that predicts whether a given trading day occurred **before or after** the implementation of TARP (October 3, 2008). The dataset includes 10-minute cumulative return data for major ETFs.

---

## 📁 Notebooks

### 1. `Market_TARP_Volatility.ipynb`

A comprehensive notebook detailing the historical, economic, and technical background of the project. It:

-   Provides context on the 2008 Global Financial Crisis and TARP
-   Breaks down the market and sector-level ETF behavior
-   Explains dataset structure and feature engineering
-   Describes the binary classification challenge: `post_TARP` (0 = pre, 1 = post)

**ETFs Analyzed**:

-   SPY (S&P 500)
-   QQQ (NASDAQ)
-   XLF (Financials)
-   XLE (Energy)
-   XLY (Consumer Discretionary)

---

### 2. `Optuna_Assignment1TaleofTwoBozos.ipynb`

This notebook uses **Optuna** for hyperparameter optimization of an XGBoost classifier.

**Best Model Results**:

-   **ROC-AUC**: `0.96196`
-   **Accuracy**: `88.57%`
-   **Baseline ROC-AUC**: `0.94746`
    -   ✅ **Improvement over baseline**: **+1.53%**
-   **Best Parameters**:
    ```json
    {
      "max_depth": 5,
      "learning_rate": 0.1638,
      "subsample": 0.8418,
      "colsample_bytree": 0.7466,
      "min_child_weight": 1,
      "gamma": 0.0696,
      "alpha": 0.2447,
      "lambda": 1.9334
    }
    ```

### 3. `Assignment1TaleofTwoBozosMaster.ipynb`

A results consolidation and feature exploration notebook that includes:

-   Volatility and volume-based feature engineering
-   ROC Curve visualization (`roc_curve.png`)
-   Final model saved to `FreeLunch9.joblib`
-   Summary DataFrame with the `post_TARP` predictions

---

## 🔬 Methodology

-   Data derived from 10-minute bar returns (preprocessed to reduce bid-ask bounce artifacts)
-   XGBoost with ROC-AUC as the primary metric
-   Cross-validation with Optuna-guided parameter tuning
-   Feature set includes time-segmented volatility, volume concentration, and cumulative return metrics

---

## 🚀 Getting Started

### Install Requirements

```bash
pip install numpy pandas matplotlib scikit-learn xgboost optuna seaborn
```
Run Jupyter Notebook
Key Insight
The model was able to distinguish pre- and post-TARP trading days with high predictive performance, suggesting measurable shifts in intraday market dynamics following government intervention — particularly in volatility and volume patterns across sectors.
