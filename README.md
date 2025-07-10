# Credit-Card-Fraud Detection

This project aims to detect fraudulent credit card transactions using various machine learning and deep learning models. It utilizes a publicly available dataset (from [Kaggle](https://www.kaggle.com/code/gpreda/credit-card-fraud-detection-predictive-models)) containing anonymized transaction features processed via 28 PCA's.

---

## Summary

Credit card fraud poses significant risks to financial institutions and customers. 
This project applies multiple classification techniques to accurately identify fraudulent transactions in highly imbalanced data.

Key highlights:
- Applied extensive **Exploratory Data Analysis (EDA)**
- Compared models on **raw** and **normalized** data wrt to Time/Amount
- Addressed data imbalance using **SMOTE** for better spread of imbalanced data
- Built a **PyTorch-based neural network** with a sigmoid function for binary classification
- Evaluated with metrics like **ROC-AUC**, **F1-score**, and **Recall** for all models.

---

##  Features
A database of a European cardholder company shared its 2-day normalised data.
The total percentage of fraud in 2 days is 0.17 out of 200K+ transactions.
No need for Normalisation as the Data itself is PCA's of Big Data.
 SMOTE oversampling to handle class imbalance  
 Raw data is Unnormalised Amount
 
 Multiple models implemented:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost
- Neural Network (PyTorch)

Metrics reporting using:
- Classification Report
- ROC AUC Score
- F1-Score & Recall
- Support

 Support for training on both **normalized** and **unnormalized** datasets  


##  Model Performance 


| Model               | ROC-AUC | F1 Score | Recall | Precision |
|--------------------|---------|----------|--------|-----------|
| Logistic Regression| 0.970   | 0.11     | 0.92   | 0.06      |
| Random Forest      | 0.926  | 0.83     | 0.83   | 0.83      |
| KNN                | 0.954   | 0.61     | 0.88   | 0.46      |
| XGBoost            | 0.933   | 0.82     | 0.85   | 0.79      |
| Neural Network     | 0.935 | 0.72     | 0.87   | 0.62      |

# Optimisations in Three Complex Models

##  Model Performance after Optimization

| Model               | Accuracy | ROC-AUC | F1 Score | Recall | Precision |
|--------------------|----------|---------|----------|--------|-----------|
| Random Forest      | 1.00     | 0.964   | 0.83     | 0.83   | 0.83      |
| XGBoost            | 1.00     | 0.982   | 0.79     | 0.87   | 0.72      |
| Neural Network     | 1.00     | 0.974   | 0.75     | 0.87   | 0.66     |

---

## ⚙️ Optimizations Applied per Model


###  Random Forest
-  `class_weight='balanced'`
-  `GridSearchCV` on `n_estimators`, `max_depth`, `min_samples_split`

###  XGBoost

- `scale_pos_weight` for imbalance handling
- `GridSearchCV` for `max_depth`, `n_estimators`, `learning_rate`

###  Neural Network Using Pytorch
-  Weighted `BCELoss` for imbalance
-  `Dropout` & `BatchNorm` to prevent overfitting
-  Threshold tuning using Precision-Recall Curve
-  Tuned layers = `3` and neurons = `129`

##Results:
Achieved high detection performance:
  - Neural Net: **F1 = 0.75**, **ROC-AUC = 0.974**
  -  XGBoost: **F1 = 0.79**, **ROC-AUC = 0.982**
  -  Random Forest: **F1 = 0.83**, **ROC-AUC = 0.964**
---

