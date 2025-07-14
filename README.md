# Code

The .py files contain the code for this project. Below is a preview about the purpose of the code in each file:

### 01_nlp_methods.py: 
Creates an extended dictionary of legal words with Glove word2vec and extracts topics via BERTopic
### 02_data_cleaning.py: 
Requires a successful run of 01_nlp such that the file agora_topic_probabilities.csv exists in the /data folder. 
### 03_run_regressions.rmd: 
Assumes cleaned_agora_inputs.csv exists and is stored in the /data folder.

# Data

The /agora and /data folder contain all the data used for this project. Below are descriptions of each folder:

## /agora
### /agora/fulltext
Contains the full text of each bill in the dataset
### documents.csv (only used .csv)
Contains the metadata about each bill which includes our dependent variable, bill enactment

## /data
### agora_raw.csv
Raw data from agora dataset. Same as documents.csv
### agora_processed.csv
Add the full text of each bill as a column to the .csv file
### agora_topic_probabilities.csv
Add the topic probabilities of each topic to the .csv file
### cleaned_agora_inputs.csv
Clean the dataset after running the `02_data_cleaning.py` file

# Usge
Please run `pip install -r requirements.txt` to download all the necessary package version necessary for this project. Afterwards, all the .py files can be ran with `python3 {01_nlp_methods | 02_data_cleaning | 03_run_regressions}.py`.
