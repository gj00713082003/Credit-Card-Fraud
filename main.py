from sklearn.model_selection import train_test_split
import pandas as pd
from model import (
    LogisticModel,
    RandomForestModel,
    KNNModel,
    XGBoostModel,
    NeuralNetModel
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE # SMOTE is used to handle class imbalance in the dataset

###################
# Load the dataset
###################
df = pd.read_csv('creditcard.csv')

# Feature/Target split
X = df.drop(['Class'], axis=1)
y = df['Class']

# Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, stratify=y, random_state=42) # This ensures that the class distribution is maintained in both training and testing sets.
# === Data Preprocessing ===
# === Normalize Time and Amount for applicable models ===
scaler = StandardScaler()
X_train_norm = X_train.copy()
X_test_norm = X_test.copy()
X_train_norm[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test_norm[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# === Raw Data for Tree-based Models ===
X_train_raw = X_train.copy()
X_test_raw = X_test.copy()

# === Apply SMOTE === #
# This is necessary to handle class imbalance in the dataset
# SMOTE will be applied to both normalized and raw datasets
# Importing SMOTE from imblearn
# SMOTE (Synthetic Minority Over-sampling Technique) is used to handle class imbalance by generating synthetic samples for the minority class.
sm = SMOTE(random_state=42)

# For normalized models
X_train_norm_res, y_train_norm_res = sm.fit_resample(X_train_norm, y_train)

# For raw models
X_train_raw_res, y_train_raw_res = sm.fit_resample(X_train_raw, y_train)

# === Run models selectively ===
# Normalized models: Logistic, SVM, KNN, Neural Net
def run_normalized_models(X_train, y_train, X_test, y_test):
    print("\n Running Normalized Models (Logistic, KNN, NN)")
    
    log_model = LogisticModel()
    log_model.train(X_train, y_train)
    log_model.evaluate(X_test, y_test)

    knn_model = KNNModel()
    knn_model.train(X_train, y_train)
    knn_model.evaluate(X_test, y_test)

    nn_model = NeuralNetModel(input_dim=X_train.shape[1])
    nn_model.train(X_train, y_train)
    nn_model.evaluate(X_test, y_test)
    

# Unnormalized models: Random Forest, XGBoost
def run_raw_models(X_train, y_train, X_test, y_test):
    print("\n Running Tree-Based Models (Random Forest, XGBoost)")

    rf_model = RandomForestModel()
    rf_model.train(X_train, y_train)
    rf_model.evaluate(X_test, y_test)

    xgb_model = XGBoostModel()
    xgb_model.train(X_train, y_train)
    xgb_model.evaluate(X_test, y_test)

# === Execute Model Training ===
run_normalized_models(X_train_norm_res, y_train_norm_res, X_test_norm, y_test) # This line is necessary to ensure the normalized models are trained on the resampled data
run_raw_models(X_train_raw_res, y_train_raw_res, X_test_raw, y_test) # This line is necessary to ensure the raw models are trained on the resampled data

# === End of Model Execution ===


