## Replication Package

All code is hosted in the main folder. Data and intermediate data is stored in and written to /data.

02_data_cleaning.py requires a successful run of 01_nlp such that the file agora_topic_probabilities.csv exists in the /data folder. 

03_run_regressions.rmd assumes cleaned_agora_inputs.csv exists and is stored in the /data folder.