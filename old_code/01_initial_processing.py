import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('oecd-ai-all-ai-policies.csv')

# Display the first few rows and basic information about the DataFrame
print("\nFirst few rows of the DataFrame:")
print(df.head())

print("\nDataFrame Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Print top 10 most common themes
print("\nTop 10 Most Common Themes:")
print(df['Theme(s)'].value_counts().head(10))
print(df['Yearly budget range'].value_counts().head(10))
print(df['Has funding from private sector ?'].value_counts().head(10))

# What leads policies to be successful?
# Yearly budget range & no cancellation reason
# Different themes? Different funding sources? Can we do NLP on the Description?