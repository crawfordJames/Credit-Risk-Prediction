# Credit Risk & Loan Default Prediction

Predicting the probability of loan default using machine learning
on the Give Me Some Credit dataset from Kaggle (150,000 records).

## Project Overview
Financial institutions lose billions annually to loan defaults.
This project builds and compares multiple classification models
to predict whether a borrower will experience financial distress
within two years using real-world credit data. The project also
incorporates SHAP explainability analysis to interpret model
decisions — a critical requirement in regulated financial environments.

## Models Compared
| Model | Type |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree | Rule-based, non-linear |
| XGBoost | Gradient boosted ensemble |
| Naive Bayes | Probabilistic baseline |

## Techniques Used
- Exploratory Data Analysis (EDA)
- Missing value imputation using median
- SMOTE for class imbalance handling (93/7 default ratio)
- Feature scaling with StandardScaler
- ROC/AUC and Precision-Recall curve evaluation
- Confusion matrix comparison across all models
- XGBoost feature importance
- SHAP explainability analysis

## Key Visualizations
- Class distribution before and after SMOTE
- Feature correlation heatmap
- Confusion matrices (all 4 models)
- ROC curves overlaid for model comparison
- Precision-Recall curves
- XGBoost feature importance bar chart
- SHAP bar and dot summary plots

## Results
XGBoost outperformed all other models across every evaluation
metric including AUC-ROC, Average Precision, Default F1, and
Default Recall. Logistic Regression performed competitively as
a linear baseline and remains a strong candidate in environments
where interpretability and regulatory compliance are priorities.

SHAP analysis confirmed that past delinquency behavior and credit
utilization are the primary drivers of default risk, consistent
with established credit scoring practices used by financial
institutions.

## Dataset
Download `cs-training.csv` from
[Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
and place it in the project root before running.

The dataset contains 150,000 records with the following features:
| Feature | Description |
|---|---|
| SeriousDlqin2yrs | Target — 1 if borrower defaulted within 2 years |
| RevolvingUtilizationOfUnsecuredLines | Credit card utilization ratio |
| Age | Age of borrower in years |
| NumberOfTime30-59DaysPastDueNotWorse | Times 30-59 days past due |
| DebtRatio | Monthly debt payments / monthly income |
| MonthlyIncome | Monthly income in dollars |
| NumberOfOpenCreditLinesAndLoans | Number of open credit lines |
| NumberOfTimes90DaysLate | Times 90+ days past due |
| NumberRealEstateLoansOrLines | Number of real estate loans |
| NumberOfTime60-89DaysPastDueNotWorse | Times 60-89 days past due |
| NumberOfDependents | Number of dependents in family |

## Requirements
```bash
pip install pandas numpy matplotlib seaborn xgboost shap imbalanced-learn scikit-learn
```

## How to Run
1. Clone the repo
```bash
git clone https://github.com/crawfordJames/Credit-Risk-Prediction.git
```
2. Navigate to the project folder
```bash
cd Credit-Risk-Prediction
```
3. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn xgboost shap imbalanced-learn scikit-learn
```
4. Download `cs-training.csv` from Kaggle and place it in the project folder
5. Open the notebook
```bash
jupyter notebook "Credit Risk & Loan Prediction.ipynb"
```
6. Run all cells from top to bottom

## Future Improvements
- Hyperparameter tuning with GridSearchCV or Optuna
- Threshold optimization to minimize False Negatives
- Adding interaction features between delinquency and utilization
- Deploying XGBoost model as a REST API using Flask or FastAPI

## Tools & Technologies
Python | Jupyter Notebook | scikit-learn | XGBoost | SHAP | 
imbalanced-learn | pandas | NumPy | Matplotlib | Seaborn

## Author
Crawford James
[GitHub](https://github.com/crawfordJames) |
[LinkedIn](https://www.linkedin.com/in/crawford-james-016043254/)
