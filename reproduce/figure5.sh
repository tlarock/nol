#!/bin/bash
# NOTE: This script relies on the NOL enviornment variable pointing to
# the highest level directory. e.g. export NOL=path/to/nol
set -ex
iterations='20'
budget='5000'
ktype=funct
k=np.log
alpha=0.01
featuretype='default'
rewardfunction='new_nodes'
savegap=0
epsilon=0.3
decay=1
burnin=0
sampling_method=node
processes=$1

declare -a data_dirs
data_dirs=(synthetic/ba-graph_N-10000_m-5_m0-5/ synthetic/N-10000_maxcc-0.95_maxgcc-0.15_avgDeg-10/)
declare -a sample_sizes
sample_sizes=(0.01 0.025 0.05 0.075 0.1)

for data_dir in "${data_dirs[@]}"
do
	base_input=${NOL}'/data/'${data_dir}/
	for sample_para in "${sample_sizes[@]}"
	do
		base_output=${NOL}'/results/'${data_dir}${sampling_method}-${sample_para}
		output_folder=${base_output}/${featuretype}-${rewardfunction}'-NOL-HTR-epsilon-'${epsilon}-decay-$decay/
		log_file=../results/logs/NOL-HTR-${sample_para}.out
		python3 ../nol/run_experiment.py -m NOL-HTR -i $base_input -s $sample_para -o $output_folder -n 1 -iter $iterations -b $budget --alpha $alpha --feats $featuretype --reward $rewardfunction --save_gap $savegap -p $epsilon --decay $decay --ktype $ktype -k $k --burn $burnin --sampling-method $sampling_method --processes $processes --log $log_file &
	done
done

wait

## PLOTTING CODE
cd ../plotting/
python figure5.py 
