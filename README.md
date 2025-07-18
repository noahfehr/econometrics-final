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

# Usage
In the project's root directory, run the command 'python3 reproduce_results.py'. This will create two files: agora_with_topic_probabilities.csv in the data/ folder which contains intermediate data and run_regressions.html in the pipeline_code/ folder which contains the results of this project. Please inspect the run_regressions.html file and compare the results to our paper. 
If there are any missing libraries, these must be installed. 
Note: the pipeline does not run the nlp_methods.py file because the results of the file are non-deterministic
