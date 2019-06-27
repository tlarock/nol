#!/bin/bash
# NOTE: This script relies on the NOL enviornment variable pointing to
# the highest level directory. e.g. export NOL=path/to/nol
set -ex

processes=$1

bash reproduce_dataset.sh 'lfr-1' $processes 'wait' &
bash reproduce_dataset.sh 'lfr-2' $processes 'wait' &
bash reproduce_dataset.sh 'lfr-3' $processes 'wait' &
bash reproduce_dataset.sh 'lfr-4' $processes 'wait' &

wait

## PLOTTING CODE
cd ../plotting/
python cumulative_reward.py 4
