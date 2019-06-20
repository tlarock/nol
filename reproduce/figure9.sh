#!/bin/bash
# NOTE: This script relies on the NOL enviornment variable pointing to
# the highest level directory. e.g. export NOL=path/to/nol
set -ex

processes=$1

bash reproduce_dataset.sh ba walk $processes 'wait' &
#bash reproduce_dataset.sh bter $processes 'wait' &
#bash reproduce_dataset.sh cora $processes 'wait' &
#bash reproduce_dataset.sh dblp $processes 'wait' &
#bash reproduce_dataset.sh caida $processes 'wait' &
#bash reproduce_dataset.sh enron $processes 'wait' &

wait

## PLOTTING CODE
cd ../plotting/
python cumulative_reward.py 9
