# Feature engineering and data preparation for machine learning

# Importing  libraries
import pandas as pd


# Loading cleaned dataset
df = pd.read_csv("../data/processed/cleaned_churn.csv")

# Converting gender & country into numerical values
df = pd.get_dummies(df, columns=["country", "gender"], drop_first=True)

# Converting encoded columns to integer type
df[['country_Germany','country_Spain','gender_Male']] = \
df[['country_Germany','country_Spain','gender_Male']].astype(int)


# Created a new feature to show relationship between balance and salary
#df['balance_salary_ratio'] = df['balance'] / (df['estimated_salary'] + 1)


# Saving processed dataset for model training

df.to_csv("../data/processed/final.csv", index=False)

print("Feature engineering and data preparation completed!")

# Loaded cleaned dataset
# Converted categorical variables into numerical form
# Created new feature for balance and salary relationship
# Saved final dataset for model training