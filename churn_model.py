churn_model
# Customer Churn Prediction Script

# ============================
# Import libraries
# ============================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# ============================
# Load dataset
# ============================

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# ============================
# Data Cleaning
# ============================

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()


# ============================
# Feature Engineering
# ============================

df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

X = pd.get_dummies(X, drop_first=True)


# ============================
# Train-Test Split
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================
# Scaling
# ============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================
# Model Training
# ============================

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)


# ============================
# Predictions
# ============================

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:,1]


# ============================
# Evaluation
# ============================

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)

print("\nAUC Score:", auc)


# ============================
# Save Model
# ============================

with open("../churn_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")