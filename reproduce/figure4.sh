#!/bin/bash

processes=$1

bash reproduce_dataset.sh 'lfr-1' $processes
bash reproduce_dataset.sh 'lfr-2' $processes
bash reproduce_dataset.sh 'lfr-3' $processes
bash reproduce_dataset.sh 'lfr-4' $processes 'wait'

wait

## PLOTTING CODE
