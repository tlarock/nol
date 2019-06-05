#!/bin/bash

processes=1

bash reproduce_dataset ba $processes
bash reproduce_dataset bter $processes
bash reproduce_dataset cora $processes
bash reproduce_dataset dblp $processes
bash reproduce_dataset enron $processes
bash reproduce_dataset caida $processes

## PLOTTING CODE
