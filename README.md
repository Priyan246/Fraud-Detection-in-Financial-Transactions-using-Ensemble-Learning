# Credit Card Fraud Detection using Explainable AI & Ensemble Methods

Welcome to the central repository for this highly accurate Credit Card Fraud Detection Machine Learning Pipeline. 

This project expands upon the foundational research presented by **Fahad Almalki** in *"Financial fraud detection using explainable AI and stacking ensemble methods."* Building on those concepts, this pipeline focuses on maximizing predictive accuracy and interpretability by utilizing advanced feature engineering alongside powerful gradient-boosted ensembles.

---

## 📌 LinkedIn Post Draft
We are excited to share our Machine Learning project: a Credit Card Fraud Detection Pipeline built on Ensemble Learning and Explainable AI.

Fraud detection in financial systems is challenging — datasets are highly imbalanced, patterns are subtle, and decisions must be interpretable. Our goal was to build a system that is both accurate and transparent. This project was developed collaboratively with Priyadarshan R and Navaneeth K.

Feature Engineering

We designed behavioral and spatial features to capture hidden patterns:

amt_zscore_global: identifies unusual transaction amounts
amt_roll_mean_3: tracks recent spending trends
txn_ratio: detects burst activity
distance_km / distance_anomaly: flags inconsistent geographic movement
Feature Selection with SHAP

We used SHAP for both interpretability and feature selection. Using 8,000 samples, we identified key drivers and removed low-value features. The most influential features were amt_zscore_global, amt_roll_mean_3, and hour, validating our approach.

Ensemble Architecture

We trained LightGBM, XGBoost, and CatBoost in parallel. Class imbalance was handled using scale_pos_weight, avoiding synthetic oversampling. Hyperparameters were optimized using Optuna across 50 trials.

Results

PR-AUC: 0.9819 | ROC-AUC: 0.9997
Precision: 95.80% | Recall: 93.65% | F1-Score: 94.71%

Cross Validation:
PR-AUC: 0.9165 | ROC-AUC: 0.9977 | Precision: 92.95% | Recall: 78.69% | F1-Score: 85.23%

To ensure robustness, we performed cross-validation using the Sparkov synthetic dataset. The consistency of SHAP-selected feature importance across two independent data distributions indicates that the model captures genuine fraud behavior rather than dataset-specific patterns.

🔗 Repository:
https://github.com/Priyan246/Fraud-Detection-in-Financial-Transactions-using-Ensemble-Learning

#MachineLearning #DataScience #FraudDetection #ExplainableAI #SHAP #EnsembleLearning #Optuna

## 🏗️ Pipeline Architecture & Feature Engineering

### 1. Project Background
This pipeline was designed to elevate accuracy benchmarks for financial fraud identification. Taking inspiration from Fahad Almalki's framework of combining Stacking/Ensemble combinations and Explainable AI (XAI), we have engineered a system that captures hidden spending anomalies in real-time.

### 2. Feature Families Set
* **Sequence & Behavior Features:** 
   * `amt_change`: Dynamic ratio comparing current transaction amounts to historical rolling means. 
   * `txn_ratio`: Burst identification computing the fraction of 24h activity taking place within the last 1h window.
* **Spatial & Distance Features:** 
   * Derived metrics utilizing the Haversine formula to compute absolute geographic travel distance (`distance_km`) versus population baselines (`distance_anomaly`).
* **Additive Smoothing:** Categorical features aggregated using global m-estimates targeting micro-categories to smooth expected values intelligently.

### 3. Imbalance Handling
Financial fraud datasets inherently suffer from massive class imbalances. To best support the structural requirements of gradient boosting architectures, imbalance was handled natively via targeted objective weighting (`scale_pos_weight`), guiding the algorithms to accurately penalize missed fraudulent events.

---

## 🤖 Models & Evaluation

### Ensemble Structure 
The core predictive capability is formed by an ensemble of three distinct gradient boosting frameworks, known for their exceptional performance on structured tabular data:
1. **LightGBM**
2. **XGBoost** 
3. **CatBoost**

Hyperparameter combinations were automatically searched and pruned efficiently using **Optuna** across a tightly parameterized GPU-oriented space to surface the most accurate configurations.

### Performance Results
Evaluation prioritized **PR-AUC** (Precision-Recall Area under Curve) as precision matters significantly in fraud domains (avoiding customer friction from false positives) alongside F1-Scores.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1-Score | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | 0.9771 | 0.9993 | 0.9690 | 0.9209 | 0.9444 | 0.719 |
| **XGBoost** | 0.9809 | 0.9997 | 0.9612 | 0.9233 | 0.9419 | 0.921 |
| **CatBoost** | 0.9760 | 0.9995 | 0.9381 | 0.9287 | 0.9334 | 0.936 |
| **Ensemble (Average)** | **0.9819** | **0.9997** | **0.9580** | **0.9365** | **0.9471** | **0.731** |

---
## Cross-Evaluation Results — Sparkov Dataset

Models trained on the primary dataset were evaluated zero-shot on the Sparkov synthetic transaction benchmark to assess generalizability across distributions.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1-Score | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | 0.8928 | 0.9969 | 0.8767 | 0.7893 | 0.8307 | 0.962 |
| **XGBoost** | 0.9184 | 0.9980 | 0.8910 | 0.8154 | 0.8515 | 0.970 |
| **CatBoost** | 0.9119 | 0.9972 | 0.8934 | 0.8163 | 0.8531 | 0.969 |
| **Ensemble (Average)** | **0.9165** | **0.9977** | **0.9295** | **0.7869** | **0.8523** | **0.967** |
## 🕵️ Explainable AI (SHAP)
Transparency is critical in financial algorithms, a core concept echoed by Fahad Almalki. **SHAP (SHapley Additive exPlanations)** was leveraged to globally translate and validate model logic using 8,000 observation extractions.

**Top Identified Decision Drivers:**
1. `amt_zscore_global`: How unusual the dollar volume was relative to global averages. 
2. `amt_roll_mean_3`: The raw momentum of immediate past transaction spend histories.
3. `hour`: Explicit fraud timing dependencies (e.g., severe localized spikes occurring outside logical regional awake times).

---

## 📸 Outputs & Visualizations
*(For the repository owner: For this README to look amazing, you should upload and embed a few visual screenshots from your notebook!)*

Please take screenshots of the output and replace the image paths below:

1. **SHAP Summary Plot:** 
`![SHAP Summary Plot](path/to/shap_summary.png)`

2. **SHAP Feature Importance Bar:** 
`![SHAP Feature Importance](path/to/shap_importance_bar.png)`

3. **Model Performance ROC/PR Curves OR Confusion Matrix:**
`![Confusion Matrix](path/to/model_comparison.png)`