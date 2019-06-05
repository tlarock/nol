#!/bin/bash

processes=1

bash reproduce_dataset 'lfr-1' $processes
bash reproduce_dataset 'lfr-2' $processes
bash reproduce_dataset 'lfr-3' $processes
bash reproduce_dataset 'lfr-4' $processes

## PLOTTING CODE
