# Feature engineering and data preparation for machine learning

# Importing libraries
import pandas as pd


# Loading cleaned dataset
df = pd.read_csv("../data/processed/cleaned_churn.csv")

#one hot encoding for country
df = pd.get_dummies(df, columns=['country','gender'],drop_first=True)

#convert boolean to int
df[['country_Germany', 'country_Spain','gender_Male']] = df[['country_Germany', 'country_Spain','gender_Male']].astype(int)

# Dropping gender column (not needed for model)
#df = df.drop(columns=["gender"])

male_count = df['gender_Male'].sum()
female_count = len(df) - male_count

# Dropping unnecessary columns
final_df = df.drop(columns=[
    'credit_score',
    'estimated_salary',
    'credit_card'
])

# Saving processed dataset for model training
final_df.to_csv("../data/processed/final.csv", index=False)

print(male_count)
print(female_count)
print("Feature engineering and data preparation completed!")


# Loaded cleaned dataset
# Dropped gender column
# Created new feature for balance and salary relationship
# Dropped unnecessary columns
# Saved final dataset for model training