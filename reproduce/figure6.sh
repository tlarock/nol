#!/bin/bash

processes=1

bash reproduce_dataset 'twitter' $processes

## PLOTTING CODE
cd ../plotting/
python cumulative_reward.py 6
