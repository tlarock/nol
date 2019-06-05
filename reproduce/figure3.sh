#!/bin/bash

processes=4

bash reproduce_dataset.sh ba $processes
bash reproduce_dataset.sh bter $processes
bash reproduce_dataset.sh cora $processes
bash reproduce_dataset.sh dblp $processes
bash reproduce_dataset.sh caida $processes
bash reproduce_dataset.sh enron $processes 'wait'
## PLOTTING CODE

wait
