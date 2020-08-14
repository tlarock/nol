#!/bin/bash
# NOTE: This script relies on the NOL enviornment variable pointing to
# the highest level directory. e.g. export NOL=path/to/nol
set -ex

# First argument to this script should be an integer
# that will determine the number of parallel processes 
# used per experiment
processes=$1

# If you want to run on all networks at once,
# just add an ampersand (&) to the end of all but the
# last line
bash reproduce_dataset.sh ba node $processes 'wait'
bash reproduce_dataset.sh bter node $processes 'wait'
bash reproduce_dataset.sh cora node $processes 'wait'
bash reproduce_dataset.sh dblp node $processes 'wait'
bash reproduce_dataset.sh caida node $processes 'wait'
bash reproduce_dataset.sh enron node $processes 'wait' 

wait

## PLOTTING CODE
cd ../plotting/
python figure3.py
