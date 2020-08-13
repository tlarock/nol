#!/bin/bash

processes=$1

bash reproduce_dataset.sh 'twitter' node $processes 'wait'

## PLOTTING CODE
cd ../plotting/
python cumulative_reward.py 6
