import pandas as pd

def remove_irrelevant_columns(df):
    # Below is a list of columns which are irrelevant, are text heavy, or are redundant given other explanatory variables available in the dataset.
    cols_to_drop = [
        'Unnamed: 0', 'AGORA ID', 'Official name', 'Casual name', 'Link to document', 'Collections',
        'full_text_preprocessed', 'full_text', 'Tags', 'Short summary', 'Long summary', 'Summaries and tags may include unreviewed machine output',
        'Official plaintext retrieved', 'Official plaintext source', 'Official plaintext unavailable/infeasible', 'Official pdf source', 'Official pdf retrieved',
        'Proposed date', 'Validated?', 'LDA_topic1','LDA_topic2', 'LDA_topic3', 'Annotated?', 'Number of segments created'
        ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    return df

def transform_authority(df):
    # Remove international authorities without US jurisdiction
    international_authorities_to_drop = [
        'Government of Israel', 'Government of New Zealand', 'Government of Canada',
        'Government of Australia', 'Government of the United Kingdom',
        'Chinese central government', 'Chinese provincial and local governments',
        'European Union', 'United Nations', 'OECD', 'Other multinational'
    ]
    df = df[~df['Authority'].isin(international_authorities_to_drop)]
    # Now, we want to divide authorities into federal legislative, federal executive, and US states
    federal_legislative = ['United States Congress']
    federal_executive = [
        'Executive Office of the President', 'Department of Defense',
        'Department of Commerce', 'Department of Agriculture',
        'Department of Health and Human Services', 'Department of Education',
        'Office of Management and Budget', 'Federal Election Commission',
        'National Institute of Standards and Technology',
        'Copyright Office, Library of Congress'
    ]
    us_states = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
        'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
        'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
        'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
        'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
        'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
        'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
        'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
        'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
        'West Virginia', 'Wisconsin', 'Wyoming', 'District of Columbia'
    ]
    # Categorizing based on lists above
    def categorize(entity):
        if entity in federal_legislative:
            return 'federal_legislative'
        elif entity in federal_executive:
            return 'federal_executive'
        elif entity in us_states:
            return entity
        else:
            return None
    df.loc[:, 'Authority'] = df['Authority'].apply(categorize)
    df = df[df['Authority'].notna()]
    # Now we will transform this column further to dilineate political tendencies of the respective states
    blue_states = [
        'California', 'Connecticut', 'Delaware', 'Hawaii', 'Illinois', 'Maine',
        'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'New Jersey',
        'New Mexico', 'New York', 'Oregon', 'Rhode Island', 'Vermont',
        'Washington', 'Colorado', 'Nevada', 'District of Columbia'
    ]

    red_states = [
        'Alabama', 'Arkansas', 'Idaho', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
        'Louisiana', 'Mississippi', 'Missouri', 'Montana', 'Nebraska',
        'North Dakota', 'Oklahoma', 'South Carolina', 'South Dakota',
        'Tennessee', 'Texas', 'Utah', 'West Virginia', 'Wyoming'
    ]

    purple_states = [
        'Arizona', 'Florida', 'Georgia', 'New Hampshire', 'North Carolina',
        'Ohio', 'Pennsylvania', 'Virginia', 'Wisconsin'  
    ]

    def color_label(entity):
        if entity in blue_states:
            return 'blue_state'
        elif entity in red_states:
            return 'red_state'
        elif entity in purple_states:
            return 'purple_state'
        else:
            return entity 

    df['Authority'] = df['Authority'].apply(color_label)
    return df

def make_dummies_drop_baselines(df):
    # We have two variables: applies to the government and applies to the private sector. We assume these to be exclusive and thus drop one as baseline.
    if 'Primarily applies to the government' in df.columns:
        df = df.drop(columns=['Primarily applies to the government'])
    # We turn our response variable into a binary outcome (from text)
    df['enacted_binary'] = (df['Most recent activity'] == 'Enacted').astype(int)
    # We can now get rid of 'Most recent activity'
    df = df.drop(columns=['Most recent activity'])
    outcome_var = df['enacted_binary']
    date_var = df['Most recent activity date']
    # Turn the remainder into dummies
    df_encoded = pd.get_dummies(df.drop(columns=['Most recent activity date', 'enacted_binary']))
    # Convert all bool columns to 0/1 integers
    for col in df_encoded.select_dtypes(include='bool').columns:
        df_encoded[col] = df_encoded[col].astype(int)
    # Concatenate back together
    df = pd.concat([df_encoded, outcome_var, date_var], axis=1)
    # Now we drop the federal_executive dummy in order to have a baseline for our analysis
    if 'Authority_federal_executive' in df.columns:
        df = df.drop(columns=['Authority_federal_executive'])
    return df

def fix_dates(df):
    # Calculate days after 2019-02-11 for each date in 'Most recent activity date'
    df['months_after_2019_02_11'] = (
        (df['Most recent activity date'].astype('datetime64[ns]') - pd.to_datetime('2019-02-11'))
        .dt.days // 30
    )
    df = df.drop(columns=['Most recent activity date'], errors='ignore')
    return df

def exclude_content_metadata(df):
    # There are many binary metadata indicators relating to the content itself. This causes colinearity issues with our BERT topics and as such we drop them. 
    exclude_prefixes = ('Strategies', 'Applications', 'Risk factors', 'Incentives', 'Harms')
    cols_to_exclude = [col for col in df.columns if col.startswith(exclude_prefixes)]
    df = df.drop(columns=cols_to_exclude, errors='ignore')
    # Sanity check to ensure that all values are numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def main():
    # Read in the intermediate data with BERT probabilities
    df = pd.read_csv('data/agora_topic_probabilities.csv')
    # Processing this data
    df = remove_irrelevant_columns(df)
    df = transform_authority(df)
    df = make_dummies_drop_baselines(df)
    df = fix_dates(df)
    df = exclude_content_metadata(df)
    # Now write this to a csv in the data folder
    df.to_csv('data/cleaned_agora_inputs.csv', index=False)

if __name__ == "__main__":
    main()




# import statsmodels.api as sm
# import numpy as np

# # Define y and X
# y = df['enacted_binary']

# # Drop y and any non-numeric columns from X
# X = df.drop(columns=['enacted_binary'])

# # Ensure all data is numeric and no object dtype remains
# X = X.apply(pd.to_numeric, errors='coerce')
# y = pd.to_numeric(y, errors='coerce')

# # # Fit a simple linear regression model
# # X_with_const = sm.add_constant(X)
# # model = sm.OLS(y, X_with_const, missing='drop')
# # results = model.fit()
# # print(results.summary())

# # # Print columns with |t| > 2 (excluding the constant)
# # t_stats = results.tvalues.drop('const', errors='ignore')
# # cols_over_2 = t_stats[abs(t_stats) > 2].index.tolist()
# # print("Columns with |t| > 2:")
# # for col in cols_over_2:
# #     print(col)

# # Run a linear regression excluding columns that start with 'Strategies' or 'Applications'

# X_no_strat_app = X.drop(columns=cols_to_exclude, errors='ignore')


# # Add constant and fit the model
# X_no_strat_app_const = sm.add_constant(X_no_strat_app)
# model_no_strat_app = sm.OLS(y, X_no_strat_app_const, missing='drop')
# results_no_strat_app = model_no_strat_app.fit()
# print("\nLinear regression excluding all of the content-related columns:")
# print(results_no_strat_app.summary())
# X_no_strat_app_with_y = X_no_strat_app.copy()
# X_no_strat_app_with_y['enacted_binary'] = y
# X_no_strat_app_with_y.