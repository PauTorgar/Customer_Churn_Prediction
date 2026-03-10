# Customer_Churn_Prediction
Customer churn prediction using Python, exploratory data analysis, and machine learning.
________________________________________________________________________________________

## Project Overview

Customer churn is one of the most important challenges for subscription-based businesses. Identifying customers who are likely to leave allows companies to take proactive actions to improve customer retention and reduce revenue loss.

This project analyzes customer behavior using the **Telco Customer Churn dataset** and builds a **machine learning model to predict customer churn**.

The analysis includes data cleaning, exploratory data analysis, feature engineering, and model training using Python.

---

## Objectives

The main objectives of this project are:

* Analyze customer behavior and identify patterns related to churn
* Perform exploratory data analysis (EDA)
* Prepare and clean the dataset for machine learning
* Build a classification model to predict customer churn
* Evaluate model performance using standard metrics
* Extract business insights from the results

---

## Dataset

The dataset used in this project is the **Telco Customer Churn Dataset**, which contains information about customers of a telecommunications company.

Dataset characteristics:

* **7032 observations**
* **21 features**

Some of the variables included in the dataset:

* Gender
* SeniorCitizen
* Partner
* Dependents
* Contract type
* Internet service
* Payment method
* Monthly charges
* Tenure
* Total charges

Target variable:

**Churn**

* Yes → Customer left the company
* No → Customer stayed with the company

---

## Data Cleaning and Preprocessing

The following preprocessing steps were performed:

* Converted `TotalCharges` to numeric format
* Removed rows containing missing values
* Dropped unnecessary columns such as `customerID`
* Converted the target variable `Churn` into binary format
* Encoded categorical variables using **One-Hot Encoding**
* Split the data into training and testing sets (80/20)

---

## Exploratory Data Analysis

Several visualizations were created to better understand customer behavior and identify patterns related to churn.

Key findings include:

* Customers with **month-to-month contracts** show the highest churn rates
* Customers paying via **electronic check** are more likely to churn
* Higher **monthly charges** are associated with increased churn probability
* Fiber optic internet users tend to churn more frequently compared to DSL users

These insights provide valuable information for potential customer retention strategies.

---

## Machine Learning Model

A **Logistic Regression** model was used to predict customer churn.

Steps performed:

1. Train-test split (80/20)
2. Feature scaling using `StandardScaler`
3. Model training
4. Model evaluation

Libraries used:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn

---

## Model Performance

Evaluation metrics obtained from the test dataset:

Accuracy:

0.78

AUC Score:

0.80

Confusion Matrix:

True Negatives: 906
False Positives: 129
False Negatives: 186
True Positives: 188

The model demonstrates good performance in identifying non-churn customers, while still providing useful predictive capability for churn cases.

---

## Important Features

Some of the features with strong influence on churn prediction include:

* StreamingMovies
* PaymentMethod_ElectronicCheck
* PaperlessBilling
* InternetService_FiberOptic
* PaymentMethod_MailedCheck

These variables highlight behavioral and billing factors that may influence customer retention.

---

## Business Insights

Based on the analysis, companies could potentially reduce churn by:

* Encouraging customers to switch to **long-term contracts**
* Promoting **automatic payment methods**
* Providing incentives for customers with **higher monthly charges**
* Improving service quality for **fiber optic internet users**

---

## Technologies Used

Python libraries used in this project:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn

---

## Project Structure

customer-churn-analysis

data/
  WA_Fn-UseC_-Telco-Customer-Churn.csv

notebooks/
  churn_analysis.ipynb

scripts/
  churn_model.py

churn_model.pkl

README.md

---

## Author

Paulina Tornero
Aspiring Data Scientist | Python | Machine Learning | Data Analysis

---

## Future Improvements

Potential improvements for this project include:

* Testing additional machine learning models (Random Forest, XGBoost)
* Hyperparameter tuning
* Feature importance analysis using SHAP values
* Deploying the model as a predictive API
