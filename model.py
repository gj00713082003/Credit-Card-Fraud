from sklearn.metrics import classification_report, roc_auc_score  # Classification report and ROC AUC score for model evaluation
from sklearn.metrics import confusion_matrix, accuracy_score  # Confusion matrix and accuracy score for model   evaluation
from sklearn.metrics import precision_recall_curve  # Precision-recall curve for model evaluation
from imblearn.over_sampling import SMOTE # SMOTE for handling class imbalance:uses the Internal KNN algorithm to generate synthetic samples for the minority class.
import numpy as np  # NumPy for numerical operations
import pandas as pd  # Pandas for data manipulation and analysis
from sklearn.linear_model import LogisticRegression # Logistic Regression for classification
from sklearn.ensemble import RandomForestClassifier # Random Forest for classification
from sklearn.svm import SVC                         # Support Vector Classifier for classification
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors for classification
from xgboost import XGBClassifier                   # XGBoost for classification
from sklearn.tree import DecisionTreeClassifier  # Decision Tree for classification
from sklearn.model_selection import GridSearchCV

## For neural network model
import torch                                     # PyTorch for building neural networks
import torch.nn as nn                            # Neural network module from PyTorch
import torch.optim as optim                      # Optimizers from PyTorch (adam/sgd)
from torch.utils.data import DataLoader, TensorDataset # DataLoader and TensorDataset for handling data in batches
## Loaded all the libraries and created the train and test data sets.
## Comparision of the models will be done on the basis of accuracy, precision, recall, f1-score and ROC AUC score.





########################################################################
#####Logistics regression#####################
class LogisticModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)  # Approx. 1000 iterations to update weights for best fit

    def train(self, X, y):
        self.model.fit(X, y)  # Fit the model to the training data

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)  # Predict class labels
        y_prob = self.model.predict_proba(X)[:, 1]  # Probability of the positive class

        print("\nLogistic Regression Evaluation")
        print(classification_report(y, y_pred))
        print("ROC AUC Score:", roc_auc_score(y, y_prob))
    # ROC AUC score is a performance measurement for classification problems at various threshold settings. It tells how much the model is capable of distinguishing between classes.

####################################################################################
### RandomForsteModel as a classification model
# This model will be used to predict the class of the data

#
####################################################################################
### RandomForsteModel as a classification model
# This model will be used to predict the class of the data


class RandomForestModel:
    def __init__(self, use_grid=False):
        self.model = None
        self.use_grid = use_grid

    def train(self, X, y):
        if self.use_grid:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced']
            }
            grid = GridSearchCV(RandomForestClassifier(random_state=42),
                                param_grid, scoring='f1', cv=3, verbose=1)
            grid.fit(X, y)
            self.model = grid.best_estimator_
            print("Best Random Forest Params:", grid.best_params_)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            self.model.fit(X, y)

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        print("\nRandom Forest")
        print(classification_report(y, y_pred))
        print("ROC AUC:", roc_auc_score(y, y_prob))
        
        
        
        

###############################################################################
class KNNModel:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=5)

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        print("\nKNN")
        print(classification_report(y, y_pred))
        print("ROC AUC:", roc_auc_score(y, y_prob))



  
###############################################################################      
##### XGBOOST ###########################

class XGBoostModel:
    def __init__(self, use_grid=False):
        self.model = None
        self.use_grid = use_grid

    def train(self, X, y):
        if self.use_grid:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'scale_pos_weight': [5, 10, 20]  # adjust based on imbalance ratio
            }
            grid = GridSearchCV(
                XGBClassifier(eval_metric='logloss', use_label_encoder=False),
                param_grid, scoring='f1', cv=3, verbose=1
            )
            grid.fit(X, y)
            self.model = grid.best_estimator_
            print("Best XGBoost Params:", grid.best_params_)
        else:
            # You may want to compute the scale_pos_weight dynamically
            self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42,
                                       scale_pos_weight=15)
            self.model.fit(X, y)

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        print("\nXGBoost")
        print(classification_report(y, y_pred))
        print("ROC AUC:", roc_auc_score(y, y_prob))


    
################################################################################
 #--- Neural Network using the PyTorch------------------------------


################################################################################
 #--- Neural Network using the PyTorch------------------------------


class NeuralNetModel:
    def __init__(self, input_dim, use_weighted_loss=True):
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),  # 1st layer
            nn.BatchNorm1d(128),        # Batch normalization for 1st layer
            nn.ReLU(),           # Activation function for 1st layer
            nn.Dropout(0.3),       # Dropout for regularization
            nn.Linear(128, 64),         # 2nd layer
            nn.BatchNorm1d(64),       # Batch normalization for 2nd layer
            nn.ReLU(),           
            nn.Dropout(0.2),         # Dropout for regularization wiht prob =0.2
            nn.Linear(64, 1),      # Output layer
            nn.Sigmoid()            # Sigmoid activation for binary classification
        )

        if use_weighted_loss:
            # Set higher weight for minority class (fraud = 1)
            self.criterion = nn.BCELoss(weight=torch.tensor([10.0]))
        else:
            self.criterion = nn.BCELoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.best_threshold = 0.5

    def train(self, X_train, y_train, epochs=15, batch_size=64):
        X_tensor = torch.tensor(X_train.values, dtype=torch.float32)        # Convert training features to PyTorch tensor
        y_tensor = torch.tensor(y_train.values.astype(np.float32)).view(-1, 1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):     # Training loop for the neural network
            epoch_loss = 0
            for xb, yb in loader:
                self.optimizer.zero_grad()  # Zero the gradients
                preds = self.model(xb)   # Forward pass
                loss = self.criterion(preds, yb) # Compute loss
                loss.backward()   # Backward pass
                self.optimizer.step()  # Update weights
                epoch_loss += loss.item() # Accumulate loss for the epoch
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        # Compute best threshold using training data
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).view(-1).numpy() # Get predictions for training data
        precision, recall, thresholds = precision_recall_curve(y_train, preds) # Calculate precision-recall curve
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        self.best_threshold = thresholds[np.argmax(f1)]
        print(f"Best Threshold: {self.best_threshold:.2f}")

    def evaluate(self, X_test, y_test):
        self.model.eval()
        X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(X_tensor).view(-1).numpy()
        y_pred = (preds >= self.best_threshold).astype(int)
        print("\nNeural Network (PyTorch)")
        print(classification_report(y_test, y_pred))
        print("ROC AUC:", roc_auc_score(y_test, preds))
# Neural Network Model
# End of Neural Network Model