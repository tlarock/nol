#!/bin/bash
# NOTE: This script relies on the NOL enviornment variable pointing to
# the highest level directory. e.g. export NOL=path/to/nol
set -ex

processes=$1

bash reproduce_dataset.sh ba node $processes 'wait' &
bash reproduce_dataset.sh bter node $processes 'wait' &
bash reproduce_dataset.sh cora node $processes 'wait'
bash reproduce_dataset.sh dblp node $processes 'wait' &
bash reproduce_dataset.sh caida node $processes 'wait' &
bash reproduce_dataset.sh enron node $processes 'wait' 

wait

## PLOTTING CODE
cd ../plotting/
python cumulative_reward.py 3
