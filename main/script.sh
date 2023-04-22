#!/bin/bash

#to run the python script as the prompt comes give the path to the file where the time series data is kept

SECONDS=0
# this command is remove the csv files created in the previous run if any:

rm phase1.csv phase1_2.csv phase1_3.csv


#command for the python code
python LSTMcodefinal.py

duration=$SECONDS
echo "$(($duration/60)) minutes and $(($duration%60)) seconds"
