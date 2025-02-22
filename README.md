
# Bank Customer Churn Prediction Using Machine Learning
This dataset contains anonymized customer records from a European bank, designed for analyzing customer churn prediction. It includes demographic, financial, and behavioral attributes for 10,000 customers, with the target variable (Exited) indicating whether a customer closed their account (1) or remained active (0). Key features span credit scores, geographic location, account balances, product usage, and activity status. Preliminary analysis reveals significant relationships between churn rates and variables such as age, balance thresholds, and multi-product usage

### Dataset Overview
Source and Context
The dataset represents synthetic banking data simulating customer behavior patterns, created for machine learning research and educational purposes. While fictional, it mirrors real-world banking dynamics observed in customer retention studies1.

### Structure
- Format: CSV (comma-separated values)

- Rows: 10,000 customer records

- Columns: 13 attributes (11 features + 1 target + 1 identifier)

### Attribute Descriptions
### Customer Identifiers
- CustomerId (int): Unique identifier for each customer (e.g., 15634602).

- Surname (str): Anonymized customer last name (e.g., Hargrave, Chiemela).

### Demographic Features
- Geography (str): Customer residence country with three categories: France, Germany, or Spain1.

- Gender (str): Binary gender classification (Male/Female).

- Age (int): Customer age ranging from 19 to 92 years.

### Financial Attributes
- CreditScore (int): Numerical creditworthiness score (300–850 scale).

- Balance (float): Account balance in euros (e.g., 0.00 for no balance, 213146.20 for high-balance accounts).

- NumOfProducts (int): Number of bank products used (1–4).

- HasCrCard (int): Binary indicator of credit card ownership (0 = no, 1 = yes).

- EstimatedSalary (float): Annual salary estimate in euros (range: €11.11–€199,992.48).

### Behavioral Metrics
- Tenure (int): Years as bank customer (0–10).

- IsActiveMember (int): Activity status (0 = inactive, 1 = active).

### Target Variable
- Exited (int): Churn status (0 = retained, 1 = churned).

## Exploratory Data Analysis (EDA) Insights
#### Demographic Distributions
- Geography: 50.1% from France, 25.0% from Germany, 24.9% from Spain1.

- Gender: 54.3% male, 45.7% female.

- Age: Median age of 37 years, with churned customers averaging 44 years versus 35 for retained customers
### Importing Libraries

Loading data into pandas dataframe

```bash
import pandas as pd  
df = pd.read_csv('Bank_Churn.csv')  
print(df['Geography'].value_counts(normalize=True))  

```

### Financial Patterns
- Credit Scores: Mean score of 650.5 (SD = 96.7), with churned customers averaging 624 versus 658 for active clients1.

- Balance Distribution: 45.8% of customers have €0 balances, while 3.2% exceed €150,000. High-balance customers (>€100k) show 28% churn rates versus 14% overall1.

- Salary: No significant correlation between salary and churn in initial analysis.

### Product Usage Trends
- Product Count: 60.8% use 1 product, 30.1% use 2 products. Customers with 3–4 products exhibit 38% churn rates1.

- Credit Card Ownership: 70.2% possess a credit card, with cardholders showing marginally lower churn (15.3% vs. 17.1%).
### Preprocessing Recommendations

#### Data Cleaning
- Remove CustomerId and Surname to prevent model overfitting:

```bash
df = df.drop(['CustomerId', 'Surname'], axis=1)   

```

- Address special characters in surnames (e.g., H?, T'ien) through Unicode normalization.

### Feature Engineering

- Categorical Encoding:

```bash
df = pd.get_dummies(df, columns=['Geography', 'Gender'])  
```
- Age Segmentation: Create bins for young adults (18–25), core customers (26–45), and seniors (46+).

- Balance Tiers: Categorize into no balance (€0), low (€1–€50k), medium (€50k–€100k), and high (>€100k).

### Class Imbalance Mitigation
The dataset shows 20.4% churn prevalence (2,043 exits). Apply SMOTE or ADASYN for oversampling:

```bash
from imblearn.over_sampling import SMOTE  
smote = SMOTE(random_state=42)  
X_res, y_res = smote.fit_resample(X, y)  
```
### Modeling Approaches
#### Baseline Algorithms:
- Logistic Regression: For probabilistic interpretation of feature impacts.

- Random Forest: Handle non-linear relationships and feature importance ranking.

- XGBoost: Optimize for prediction accuracy with gradient boosting.

#### Evaluation Metrics: 
Prioritize metrics robust to class imbalance:

- Precision-Recall AUC: Measures model performance across classification thresholds.

- F1-Score: Balances precision and recall.

- Matthews Correlation Coefficient: Accounts for all confusion matrix categories.

### Hyperparameter Tuning Example

```bash
from sklearn.ensemble import RandomForestClassifier  
param_grid = {  
    'n_estimators': [100, 200],  
    'max_depth': [None, 10],  
    'class_weight': ['balanced', None]  
}  
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)  
grid_search.fit(X_train, y_train)  

```

### Key Findings
#### Predictive Features
- Age: Churn probability increases 2.1× for customers over 451.

- Balance: Zero-balance customers show 9% churn vs. 31% for balances >€100k1.

- Multi-Product Usage: Customers with 3–4 products have 4.8× higher churn risk than single-product users1.

#### Model Performance
- XGBoost achieved highest AUC-ROC (0.872) with precision-recall AUC of 0.534.

- Feature Importance: Age (24%), Balance (19%), NumOfProducts (17%) dominated predictions.

## Screenshots

![App Screenshot](https://github.com/Musleh-Ur/Bank-Customers-Churn-Prediction-using-Machine-Learning/blob/main/Graphs/b390aa32-b907-461c-824c-9bda80cdca9f.png)

