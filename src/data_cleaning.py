# Data cleaning and preprocessing performed on raw dataset

# Importing required libraries
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Loading raw dataset
df = pd.read_csv("data/raw/Bank Customer Churn Prediction.csv")

# Checking duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Removing duplicates
df.drop_duplicates(inplace=True)

# Checking datatypes
df.dtypes

# Dropping customer_id (not useful for prediction)
df.drop("customer_id", axis=1, inplace=True)



# Finding outliers in balance
Q1 = df['balance'].quantile(0.25)
Q3 = df['balance'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['balance'] < lower_bound) | (df['balance'] > upper_bound)]
print(len(outliers))

# Boxplot for balance
sns.boxplot(x=df['balance'])
plt.title("Balance Boxplot")
plt.show()

# Balance column has no outliers, so no action needed


# Finding outliers in age
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['age'] < lower_bound) | (df['age'] > upper_bound)]
print(len(outliers))

# Boxplot for age
sns.boxplot(x=df['age'])
plt.title("Age Boxplot")
plt.show()

# Age is an important feature, so outliers are not removed


# Finding outliers in credit_score
Q1 = df['credit_score'].quantile(0.25)
Q3 = df['credit_score'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['credit_score'] < lower_bound) | (df['credit_score'] > upper_bound)]
print(len(outliers))

# Boxplot for credit_score
sns.boxplot(x=df['credit_score'])
plt.title("Credit Score Boxplot")
plt.show()

# Credit_score has outliers but no handling is applied


# Finding outliers in estimated_salary
Q1 = df['estimated_salary'].quantile(0.25)
Q3 = df['estimated_salary'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['estimated_salary'] < lower_bound) | (df['estimated_salary'] > upper_bound)]
print(len(outliers))

# Boxplot for estimated_salary
sns.boxplot(x=df['estimated_salary'])
plt.title("Estimated Salary Boxplot")
plt.show()

# Estimated_salary has no significant outliers, so no action needed


# Saving cleaned dataset
df.to_csv("data/processed/cleaned_churn.csv", index=False)

print("Processed dataset saved successfully!")