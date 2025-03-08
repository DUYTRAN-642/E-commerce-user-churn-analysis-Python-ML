# E-commerce-user-churn-analysis-Python-ML
Predict churned users in an e-commerce platform
Certainly! Below is a more detailed version of the `README` file, including the steps for coding, an explanation of the methodology, and some example code snippets for each part of the process. I have incorporated more context and explanation for each section of the project to help users understand how to implement the solution:

---

# Churn Prediction for E-commerce Company

This project focuses on predicting churned users in an e-commerce platform in order to offer potential promotions. By analyzing customer behavior, building a machine learning model for churn prediction, and segmenting churned users into groups, we can provide targeted promotions to retain users and reduce churn.

## Dataset

The dataset, `churn_predict.csv`, contains various customer-related features and a churn flag indicating whether a customer has churned or not. Below is a description of the columns in the dataset:

| **Variable Name**          | **Description**                                         |
|----------------------------|---------------------------------------------------------|
| `CustomerID`               | Unique customer ID                                      |
| `Churn`                     | Churn flag (1 if churned, 0 if not churned)             |
| `Tenure`                    | Tenure of the customer in the organization              |
| `PreferredLoginDevice`      | Preferred login device of the customer                  |
| `CityTier`                  | City tier (1, 2, or 3)                                  |
| `WarehouseToHome`           | Distance between the warehouse and customer's home      |
| `PreferPaymentMethod`       | Preferred payment method of the customer                |
| `Gender`                    | Gender of the customer                                  |
| `HourSpendOnApp`            | Number of hours spent on the app or website             |
| `NumberOfDeviceRegistered`  | Total number of devices registered by the customer      |
| `PreferedOrderCat`          | Preferred order category of the customer in the last month |
| `SatisfactionScore`         | Customer's satisfaction score on the service            |
| `MaritalStatus`             | Marital status of the customer                          |
| `NumberOfAddress`           | Total number of addresses registered by the customer    |
| `Complain`                  | Whether the customer has raised a complaint in the last month |
| `OrderAmountHikeFromLastYear` | Percentage increase in orders from last year          |
| `CouponUsed`                | Total number of coupons used by the customer in the last month |
| `OrderCount`                | Total number of orders placed by the customer in the last month |
| `DaySinceLastOrder`         | Number of days since the customer's last order          |
| `CashbackAmount`            | Average cashback received by the customer in the last month |

## Objectives

This project addresses the following questions:
1. **Churn Behavior Analysis**: What are the patterns/behaviors of churned users? Based on this analysis, what are your suggestions to the company to reduce churned users?
2. **Churn Prediction**: Build a machine learning model to predict churned users, including fine-tuning the model for better performance.
3. **Segmentation of Churned Users**: Based on the behaviors of churned users, segment them into different groups for targeted promotions. What are the key differences between these groups?

## Approach

### 1. Churn Behavior Analysis

To come up with solutions for understanding the patterns of churned users, the following steps are performed:

- **Data Exploration**: The first step is to explore the dataset to find significant patterns, correlations, and trends between user behavior and churn. For example:
   - Users with low satisfaction scores and high complaint rates may have a higher likelihood of churning.
   - Analyze features like `SatisfactionScore`, `Complain`, `OrderCount`, and `OrderAmountHikeFromLastYear` to see if they correlate with the churn flag.

#### Code to load the dataset and explore basic statistics:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("churn_predict.csv")

# Show basic statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Visualizing churn distribution
sns.countplot(data['Churn'])
plt.title('Churn Distribution')
plt.show()
```

- **Feature Analysis**: We then analyze individual features like `SatisfactionScore`, `Complain`, `HourSpendOnApp`, `OrderAmountHikeFromLastYear`, and others that could have a strong influence on the churn flag. Correlation analysis and visualizations like pair plots can help in identifying relationships.

#### Code for visualizing feature correlations:

```python
# Correlation heatmap
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
```

### 2. Churn Prediction Model

Once the data exploration is done, we proceed to build a churn prediction model using machine learning techniques. The following steps are followed:

- **Data Preprocessing**:
   - Handle missing values if any (e.g., using imputation).
   - Encode categorical variables (`Gender`, `PreferredLoginDevice`, etc.).
   - Scale numerical features if required.

#### Code to preprocess data:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Handle missing values (if any)
data.fillna(data.mean(), inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Split data into features (X) and target (y)
X = data.drop('Churn', axis=1)
y = data['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

- **Model Selection**: We will train multiple models like Logistic Regression, Random Forest, and XGBoost to predict churn. We will also evaluate model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

#### Code to train and evaluate models:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
print("Random Forest Model Evaluation:")
print(classification_report(y_test, y_pred_rf))
```

- **Model Fine-Tuning**: Fine-tune the models using GridSearchCV to optimize hyperparameters for better performance.

#### Code to fine-tune model using GridSearchCV:

```python
from sklearn.model_selection import GridSearchCV

# Hyperparameters for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# GridSearchCV for Random Forest
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters for Random Forest:", grid_search.best_params_)
```

### 3. Segmentation of Churned Users

Once the churn prediction model is built and fine-tuned, we proceed to segment churned users into different groups based on their behaviors. This can be done using clustering techniques such as K-Means, DBSCAN, or hierarchical clustering.

#### Code for K-Means clustering:

```python
from sklearn.cluster import KMeans

# Use features like satisfaction score, complaint, order amount hike for clustering
X_cluster = data[['SatisfactionScore', 'Complain', 'OrderAmountHikeFromLastYear', 'HourSpendOnApp']]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_cluster)

# Add cluster labels to the data
data['Cluster'] = clusters

# Visualizing the clusters
sns.scatterplot(x='SatisfactionScore', y='HourSpendOnApp', hue='Cluster', data=data, palette='Set1')
plt.title("Churned User Segmentation")
plt.show()
```

#### Segment Analysis:
- After clustering, analyze each cluster to determine the common characteristics of the users in each group (e.g., high-value users, frequent complainers).
- Suggest promotions for each group based on the cluster's behavior (e.g., for "high-value" users, loyalty rewards; for "frequent complainers", support and satisfaction improvement).

### Requirements

To run this project, you will need:
* Python 3.x
* Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `statsmodels`, `plotly` (for visualizations)
* Jupyter Notebook (or Google Colab) to run the `.ipynb` file.

### Instructions to Run

1. Clone this repository or download the `.ipynb` file.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter notebook file in your preferred IDE or Google Colab.
4. Run the cells sequentially to explore the data, train the model, and perform the segmentation.
