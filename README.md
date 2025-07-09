# Credit-Card-Fraud Detection

This project aims to detect fraudulent credit card transactions using various machine learning and deep learning models. It utilizes a publicly available dataset (from Kaggle) containing anonymized transaction features processed via PCA.

---

## Summary

Credit card fraud poses significant risks to financial institutions and customers. This project applies multiple classification techniques to accurately identify fraudulent transactions in highly imbalanced data.

Key highlights:
- Applied extensive **Exploratory Data Analysis (EDA)**
- Compared models on **raw** and **normalized** data
- Addressed data imbalance using **SMOTE**
- Built a **PyTorch-based neural network**
- Evaluated with metrics like **ROC-AUC**, **F1-score**, and **Recall**

---

##  Features

 Clean and modular code structure  
 Preprocessing pipeline with PCA-aware normalization  
 SMOTE oversampling to handle class imbalance  
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
| Random Forest      | 0.964   | 0.83     | 0.83   | 0.83      |
| KNN                | 0.954   | 0.61     | 0.88   | 0.46      |
| XGBoost            | 0.983   | 0.82     | 0.85   | 0.79      |
| Neural Network     | 0.935+ | 0.72     | 0.87   | 0.62      |

# Optimisations in Three Complex Models
## 📊 Model Performance

| Model               | ROC-AUC | F1 Score | Recall | Precision |
|--------------------|---------|----------|--------|-----------|
| Logistic Regression| 0.970   | 0.11     | 0.92   | 0.06      |
| Random Forest      | 0.964   | 0.83     | 0.83   | 0.83      |
| KNN                | 0.954   | 0.61     | 0.88   | 0.46      |
| XGBoost            | 0.983   | 0.82     | 0.85   | 0.79      |
| Neural Network     | ~0.935  | 0.72     | 0.87   | 0.62      |
| SVM                | TBD     | TBD      | TBD    | TBD       |

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
-  Dropout` & `BatchNorm` to prevent overfitting
-  Threshold tuning using Precision-Recall Curve
-  Tuned layers and neurons





---

