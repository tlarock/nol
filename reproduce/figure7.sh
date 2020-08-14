#!/bin/bash
# NOTE: This script relies on the NOL enviornment variable pointing to
# the highest level directory. e.g. export NOL=path/to/nol
set -ex

# First argument to this script should be an integer
# that will determine the number of parallel processes 
# used per experiment
processes=$1

bash reproduce_dataset.sh er $processes 'wait' &
bash reproduce_dataset.sh regular $processes 'wait' &

wait

## PLOTTING CODE
cd ../plotting/
python figure7.py
