#!/bin/bash

# First argument to this script should be an integer
# that will determine the number of parallel processes 
# used per experiment
processes=$1

bash reproduce_dataset.sh 'twitter' node $processes 'wait'

## PLOTTING CODE
cd ../plotting/
python figure4.py
