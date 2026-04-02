# Feature engineering and data preparation for machine learning

# Importing libraries
import pandas as pd
import numpy as np

# Loading cleaned dataset
df = pd.read_csv("data/processed/cleaned_churn.csv")

# Dropping gender column (not needed for model)
df = df.drop(columns=["gender"])

# Converting country into numerical values
df = pd.get_dummies(df, columns=["country"], drop_first=True)

# Converting encoded columns to integer type
df[['country_Germany','country_Spain']] = \
df[['country_Germany','country_Spain']].astype(int)

# Created a new feature to show relationship between balance and salary
df['balance_salary_ratio'] = df['balance'] / (df['estimated_salary'] + 1)

# Dropping unnecessary columns
final_df = df.drop(columns=[
    'credit_score',
    'estimated_salary',
    'credit_card'
])

# Saving processed dataset for model training
final_df.to_csv("data/processed/final.csv", index=False)

print("Feature engineering and data preparation completed!")


# Loaded cleaned dataset
# Dropped gender column
# Converted categorical variable (country) into numerical form
# Created new feature for balance and salary relationship
# Dropped unnecessary columns
# Saved final dataset for model training