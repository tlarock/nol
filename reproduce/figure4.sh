#!/bin/bash

processes=$1

bash reproduce_dataset.sh 'twitter' node $processes 'wait'

## PLOTTING CODE
cd ../plotting/
python figure4.py
