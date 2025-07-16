from data_cleaning import main
import pandas as pd
import subprocess


# import the result of the nlp_methods.py file since it gives a non-deterministic result
df = pd.read_csv("data/agora_with_topic_probabilities.csv")
df_final = main(df)

# save the csv so that it can be used in the R file below
df_final.to_csv("data/cleaned_agora_inputs.csv", index=False)

# Run R code which reads the csv file saved above and creates the html file which contains the results of this paper
r_command = "options(repos = c(CRAN = 'https://cloud.r-project.org')); rmarkdown::render('run_regressions.rmd')"
subprocess.run(["Rscript", "-e", r_command])
