This file contains a complete Google Colab notebook for a Loan Status Prediction project using Machine Learning in Python.
The workflow follows a structured data science process, from importing and exploring data to building, evaluating, and optimising predictive models.

Overview of Steps

Importing Libraries:
Uses key Python libraries including NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn for data processing, visualisation, and modelling.

Data Loading & Exploration:
Loads a dataset (loan_data.csv) from Google Drive containing 45,000 records with variables such as person_age, person_income, loan_amnt, credit_score, and loan_status.
Includes descriptive statistics and data type inspection.

Data Cleaning & Preparation:

Drops irrelevant columns (person_gender, loan_intent, etc.)

Encodes categorical features numerically.

Splits data into features (X) and target (y).

Exploratory Data Analysis:

Generates correlation heatmaps.

Visualises feature distributions using histograms.

Data Splitting & Scaling:
Divides data into training and testing sets (80/20) and standardises numeric features using StandardScaler.

Model Building & Evaluation:
Trains and compares three algorithms:

Logistic Regression (Accuracy: 0.89)

Random Forest Classifier (Accuracy: 0.91)

Gradient Boosting Classifier (Accuracy: 0.91)
Outputs precision, recall, and F1-scores for each.

Model Enhancement:

Applies K-Fold Cross-Validation to improve reliability (average accuracy ≈ 91.1%).

Analyses feature importance using Random Forests.

Suggests evaluating beyond accuracy metrics.

Visualisation:
Bar charts display model comparison and feature importance rankings.

Purpose

The notebook aims to predict whether a loan will be approved or not based on borrower attributes and loan characteristics. It provides a baseline comparison of classification models for financial risk assessment.
