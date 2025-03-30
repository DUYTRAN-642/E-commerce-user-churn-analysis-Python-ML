# 📊 Project Title: Churn Prediction for E-commerce Company
Predict churned users in an e-commerce platform
Certainly! Below is a more detailed version of the `README` file, including the steps for coding, an explanation of the methodology, and some example code snippets for each part of the process. I have incorporated more context and explanation for each section of the project to help users understand how to implement the solution

![image](https://github.com/user-attachments/assets/efa5f9ca-50d0-4eb8-a23f-5d2b3222b262)

---
# 📑 Table of Contents

[📌 Background & Overview](#-background--overview)

[📂 Dataset Description & Data Structure](#-dataset-description--data-structure)

[🔎 Final Conclusion & Recommendations](#-final-conclusion--recommendations)

## 📌 Background & Overview

This project focuses on predicting churned users in an e-commerce platform in order to offer potential promotions. By analyzing customer behavior, building a machine learning model for churn prediction, and segmenting churned users into groups, we can provide targeted promotions to retain users and reduce churn.

This project addresses the following questions:
1. **Churn Behavior Analysis**: What are the patterns/behaviors of churned users? Based on this analysis, what are your suggestions to the company to reduce churned users?
2. **Churn Prediction**: Build a machine learning model to predict churned users, including fine-tuning the model for better performance.
3. **Segmentation of Churned Users**: Based on the behaviors of churned users, segment them into different groups for targeted promotions. What are the key differences between these groups?

👤 Who is this project for?

 ✔️ Marketing Teams: Understanding churn patterns can help in crafting more personalized marketing campaigns and retention offers to keep customers engaged.
 
 ✔️ Product Managers: They can use churn prediction to understand what parts of the product experience might be driving users to leave and work to improve these areas.
 
 ✔️ This project is for Data Analytics Teams who will build, train, and deploy the churn prediction model, as well as interpret the results for actionable business insights.

## 📂 Dataset Description & Data Structure

### 📌 Data Source

Source: The dataset was obtained from an e-commerce company | Source link: https://docs.google.com/spreadsheets/d/1U_M98PksVuLXT19dPORYsowH6a3mblQT/edit?gid=216773957#gid=216773957

Size: There is one table includes over 5600 rows and 20 features about customers' information 

Format: .xlsx

### 📊 Data Structure 

👉🏻 The dataset, `churn_predict.csv`, contains various customer-related features and a churn flag indicating whether a customer has churned or not. Below is a description of the columns in the dataset:

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

## ⚒️ Main Process

### 1. Churn Behavior Analysis

To come up with solutions for understanding the patterns of churned users, the following steps are performed:

- **Data Exploration**:
  * Check missing values
  ```python
  df.info()
  df.shape
  ```

![image](https://github.com/user-attachments/assets/3508ad69-61ec-44f5-98a3-47ab4dfacb52)

=> there are 6/20 columns have missing values which can cause incomplete data, lose important information, and the model's ability to generalize to unseen data could suffer. 

  * The missing values part was handled by using `KNNImputer` (K-Nearest Neighbors Imputer) which has several strong advantages, especially when compared to simpler imputation methods like mean, median, or mode imputation
```python
from sklearn.impute import KNNImputer
mis_cols =['Tenure','WarehouseToHome','HourSpendOnApp','OrderAmountHikeFromlastYear','CouponUsed','OrderCount','DaySinceLastOrder']

numerical_cols = df.select_dtypes(include=np.number).columns.tolist() # Select numerical columns
# Ensure that mis_cols only includes numerical columns
mis_cols = list(set(mis_cols) & set(numerical_cols))


imputer = KNNImputer(n_neighbors=2)
df[mis_cols] = imputer.fit_transform(df[mis_cols])
```
* Replace duplicates category in the features
```python
df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace('Mobile Phone', 'Phone')
df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace('CC', 'Credit Card')
df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace('COD', 'Cash on Delivery')
```
* Explore the dataset to find significant patterns, correlations, and trends between user behavior and churn by applying `correlation`
Users with short `Tenure` and high `Complain` rates tend to have moderate relationship with churning
```python
corr = num_cols.corr()
sns.heatmap(corr, annot=True, fmt=".1f", cmap='coolwarm', linewidths=.7)
```

![image](https://github.com/user-attachments/assets/00187ddd-3f77-4b98-8f81-ee85f2198fdd)


   - Analyze others features like `SatisfactionScore`, `OrderCount`, and `OrderAmountHikeFromLastYear`, ... not have the correlation with churning

### 2. Churn Prediction Model

Once the data exploration is done, we proceed to build a churn prediction model using machine learning techniques. The following steps are followed:

**Model Selection**: We will train model XGBoost to predict churn. We will also evaluate model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
  * Apply train_test_split
    
  ```python
  from sklearn.model_selection import train_test_split
x=df_drop.drop('Churn', axis = 1)
y=df_drop[['Churn']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
```

* Normalize for `x_test` and `x_train` seperately to avoid data leakage
```python
Normalize data
from sklearn.preprocessing import MinMaxScaler

Scale Feature:
scaler = MinMaxScaler()
model=scaler.fit(x_train)
scaled_data_train = model.transform(x_train)
scaled_data_test = model.transform(x_test)

Create DataFrames 
scaled_df_train = pd.DataFrame(scaled_data_train, columns=x_train.columns)
scaled_df_test = pd.DataFrame(scaled_data_test, columns=x_test.columns)
```

* Run the model and do evaluation
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
params = {
    'objective': 'binary:logistic',  # Binary classification
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'eval_metric': 'auc'  # Area Under ROC Curve
}
model = xgb.XGBClassifier(**params)
model.fit(x_train, y_train)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

![image](https://github.com/user-attachments/assets/42bebb78-9976-4f0e-96a0-6e0ac8c6eb3b)


==> This is a strong model overall with excellent discriminative ability (ROC-AUC 0.94) and high accuracy (0.92)

### 3. Segmentation of Churned Users

Once the churn prediction model is built and fine-tuned, we proceed to segment churned users into different groups based on their behaviors. This can be done using clustering techniques  `K-Means`

* Using Business domain and Elbow Method identify the number of cluster K = 4

![image](https://github.com/user-attachments/assets/5f7742f0-f9fa-4b01-b5e8-2a3fc1e3b3a6)

* Code for K-Means clustering:
```python
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

#Add clusters to data
df_seg['Cluster'] = clusters
#print(df_seg.groupby('Cluster').mean())  # Centroids
df_seg.head()
```

***Because there are several features in each clusters which is difficult to segment churned useres into groups, one more model `Randomforest` was using on the churend dataset (label = 0) to find feature importance***

Features `CashbackAmount`,`PreferedOrderCat` was selected based on running model and features `Tenure`, `daySinceLastOrder` was also chosen because this is e-commerce business

Calculate mean of each feature to specify the differences.

#### Segment Analysis:

* After clustering, analyze each cluster to determine the common characteristics of the users in each group (e.g., high-value users, frequent complainers).
* Suggest promotions for each group based on the cluster's behavior (e.g., for "high-value" users, loyalty rewards; for "frequent complainers", support and satisfaction improvement).
Based on KMeans Clusterring and business domain, certain number of features selected and do segmentation analysis

| **Cluster**                        | **Description**                                                                 | **Action**                                                                                       |
|------------------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| **Cluster 0 (Loyal but Inactive Users)** | Long tenure, highest cashback, but not very active recently. Diverse product preferences (others category). | ➡️ Focus on re-engagement with targeted offers in niche categories.                                |
| **Cluster 1 (Recent, Mobile-Focused Users)** | New users with very recent purchases, moderate cashback. Strong preference for mobile products. | ➡️ Engage with mobile-related promotions to nurture loyalty.                                        |
| **Cluster 2 (New, Mobile-Focused Users)** | Recent users with a strong preference for mobile phones.                         | ➡️ Offer tailored promotions for mobile products to retain engagement.                             |
| **Cluster 3 (Diverse Interests, Moderate Activity)** | Moderate tenure, decent cashback, but moderate inactivity. Diverse product preferences, leaning towards fashion and laptops. | ➡️ Provide cross-category offers (e.g., fashion and laptops) to boost engagement.                  |

## 🔎 Final Conclusion & Recommendations

👉🏻 The E-commerce User Churn Analysis project provides valuable insights into user behavior within the e-commerce space using machine learning techniques. By analyzing historical user data, we successfully identified key factors contributing to user churn and developed predictive model XGBoost to forecast potential churners

👉🏻 Identified 4 typical group of customers whose behaviors tend to lead to churning, which help stakeholders easy to define thos customers at the early stage and have the appropriate strategies for specific segments and finally save customer from leaving their ecommerce business.

📌 Key Takeaways:

✔️ The following recommendations can help businesses reduce churn rates and improve customer retention: Focus on Mobile Promotions; Cross-Category Offers; Improve Customer Experience

✔️ Model Optimization: Continuously update models with new data for better predictions.

✔️ Google Colab link: https://colab.research.google.com/drive/1XOGaCOb1VHH5zpURvlgGqUyAHy5fJrxt
