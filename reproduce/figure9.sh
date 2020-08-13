#!/bin/bash
# NOTE: This script relies on the NOL enviornment variable pointing to
# the highest level directory. e.g. export NOL=path/to/nol
set -ex

processes=$1

bash reproduce_dataset.sh ba walk $processes 'wait' & 
bash reproduce_dataset.sh bter walk $processes 'wait' &
bash reproduce_dataset.sh cora walk $processes 'wait'
bash reproduce_dataset.sh dblp walk $processes 'wait' &
bash reproduce_dataset.sh caida walk $processes 'wait' &
bash reproduce_dataset.sh enron walk $processes 'wait' 

wait

## PLOTTING CODE
cd ../plotting/
python figure9.py
