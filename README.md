# E-commerce-user-churn-analysis-Python-ML
Predict churned users in an e-commerce platform
Certainly! Below is the README file with appropriate markdown formatting using `#` and `*`:

---

# Churn Prediction for E-commerce Company

This project aims to predict churned users in an e-commerce platform in order to offer potential promotions. We will analyze customer behavior, build a machine learning model for churn prediction, and segment churned users into groups for targeted promotions.

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
   * **Data Exploration**: Analyze the dataset to identify significant patterns, correlations, and trends between user behavior and churn. For example, users with low satisfaction scores, fewer orders, and high complaint rates may be more likely to churn.
   * **Feature Analysis**: Identify the key features that might influence churn, such as `SatisfactionScore`, `Complain`, `HourSpendOnApp`, and `OrderAmountHikeFromLastYear`.
   * **Segmentation of Churned Users**: Explore the possibility of segmenting churned users based on behavioral patterns to understand distinct groups that have similar characteristics (e.g., high complaint rates, low satisfaction scores).

**Suggestions to reduce churn**:
   * Improve customer satisfaction by addressing common complaints.
   * Target users with personalized promotions based on their preferences (`PreferredOrderCat`, `PreferPaymentMethod`).
   * Offer loyalty rewards or discounts to high-spending users to incentivize retention.

### 2. Churn Prediction Model

To come up with solutions for predicting churned users, the following steps are performed:
   * **Data Preprocessing**: Clean the data by handling missing values, encoding categorical variables, and scaling numerical features.
   * **Model Selection**: Train multiple machine learning models (e.g., Logistic Regression, Random Forest, Gradient Boosting) to predict the churn flag (`Churn`).
   * **Model Evaluation**: Evaluate the models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC. Select the best model based on performance.
   * **Model Fine-tuning**: Use hyperparameter optimization techniques like GridSearchCV or RandomizedSearchCV to fine-tune the model for better performance.

**Suggested Models**:
   * **Logistic Regression**: Simple model for baseline prediction.
   * **Random Forest Classifier**: More complex model that can handle non-linear relationships and interactions.
   * **XGBoost**: Gradient boosting model that provides great performance for classification tasks.

### 3. Segmentation of Churned Users

To come up with solutions for segmenting churned users into groups, the following steps are performed:
   * **Clustering Analysis**: Apply clustering algorithms like K-Means or DBSCAN to segment churned users based on their behaviors and characteristics.
   * **Feature Selection**: Select the most important features for clustering, such as `SatisfactionScore`, `OrderAmountHikeFromLastYear`, `NumberOfDeviceRegistered`, `DaySinceLastOrder`, and `Complain`.
   * **Cluster Evaluation**: Evaluate and analyze the clusters to identify groups with distinct characteristics (e.g., "high-value users", "frequent complainers", "low spenders").
   * **Interpretation**: Visualize the clusters using techniques like PCA (Principal Component Analysis) or t-SNE to better understand the grouping.

**Segmentation Recommendations**:
   * **Group 1**: High-value, low complaint users who may benefit from loyalty promotions.
   * **Group 2**: Low-value, high complaint users who need targeted support and customer satisfaction improvements.
   * **Group 3**: Mid-value users who may need promotions or incentives to prevent churn.

## Requirements

To run this project, you will need:
* Python 3.x
* Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `statsmodels`, `sklearn`, `plotly` (for visualizations)
* Jupyter Notebook (or Google Colab) to run the `.ipynb` file.

