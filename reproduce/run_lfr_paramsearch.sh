#!/bin/bash
# NOTE: This script relies on the NOL enviornment variable pointing to
# the highest level directory. e.g. export NOL=path/to/nol

mu_vals="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7"
alpha_vals="2 2.25 2.5 2.75 3 3.25 3.5"

for mu in $mu_vals
do
	for alpha in $alpha_vals
	do
		# lfr_paramsearch.sh relies on $mu 
		# and $alpha vals set in this loop
		wait bash lfr_paramsearch.sh
	done
done
