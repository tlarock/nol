#!/bin/bash
# NOTE: This script relies on the NOL enviornment variable pointing to
# the highest level directory. e.g. export NOL=path/to/nol
set -ex

dataset=$1
sample_para=0.01
iterations='20'
budget='5000'
alpha=0.01
featuretype='default'
rewardfunction='new_nodes'
savegap=0
epsilon=0.3
decay=1
burnin=0
sampling_method=$2
processes=$3

case ${dataset} in 'ba')
	data_dir='synthetic/ba-graph_N-10000_m-5_m0-5/';;
'bter')
	data_dir='synthetic/N-10000_maxcc-0.95_maxgcc-0.15_avgDeg-10/';;
'er')
	data_dir='synthetic/er-graph_N-10000_p-0.001/';;
'lfr-1')
	data_dir='synthetic/lfr-graph_N-34546_mu-0.1/';;
'lfr-2')
	data_dir='synthetic/lfr-graph_N-34546_mu-0.2/';;
'lfr-3')
	data_dir='synthetic/lfr-graph_N-34546_mu-0.3/';;
'lfr-4')
	data_dir='synthetic/lfr-graph_N-34546_mu-0.4/';;
'dblp')
	data_dir='dblp/'
	budget=4000;;
'cora')
	data_dir='cora/';;
'caida')
	data_dir='caida/';;
'enron')
	data_dir='enron/';;
'regular')
	data_dir='regular/';;
'twitter')
	data_dir='twitter/'
	budget=50000;;
esac

base_input=${NOL}'/data/'${data_dir}
base_output=${NOL}'/results/'${data_dir}${sampling_method}-${sample_para}

## NOL(\epsilon=0.3)
output_folder=${base_output}/${featuretype}-${rewardfunction}'-NOL-epsilon-'${epsilon}-decay-$decay/
log_file=../results/logs/NOL-${dataset}.out

python3 ../nol/run_experiment.py -m NOL -i $base_input -s $sample_para -o $output_folder -n 1 -iter $iterations -b $budget --alpha $alpha --feats $featuretype --reward $rewardfunction --save_gap $savegap -p $epsilon --decay $decay --burn $burnin --sampling-method $sampling_method --processes $processes --log $log_file 

## NOL-HTR(\epsilon=0.3, k=ln(n))
ktype=funct
k=np.log
output_folder=${base_output}/${featuretype}-${rewardfunction}'-NOL-HTR-epsilon-'${epsilon}-decay-$decay/
log_file=../results/logs/NOL-HTR-${dataset}.out
python3 ../nol/run_experiment.py -m NOL-HTR -i $base_input -s $sample_para -o $output_folder -n 1 -iter $iterations -b $budget --alpha $alpha --feats $featuretype --reward $rewardfunction --save_gap $savegap -p $epsilon --decay $decay --ktype $ktype -k $k --burn $burnin --sampling-method $sampling_method --processes $processes --log $log_file 

## Random
output_folder=${base_output}/baseline-${rewardfunction}'-rand/'
log_file=../results/logs/rand-${dataset}.out
python3 ../nol/run_experiment.py -m rand -i $base_input -s $sample_para -o $output_folder -n 1 -iter $iterations -b $budget --reward $rewardfunction --save_gap $savegap --sampling-method $sampling_method --processes $processes --log $log_file 

## High
output_folder=${base_output}/baseline-${rewardfunction}'-high/'
log_file=../results/logs/high-${dataset}.out
python3 ../nol/run_experiment.py -m high -p 0 -i $base_input -s $sample_para -o $output_folder -n 1 -iter $iterations -b $budget --reward $rewardfunction --save_gap $savegap --sampling-method $sampling_method --processes $processes --log $log_file 

## High-jump
output_folder=${base_output}/baseline-${rewardfunction}'-high-jump/'
log_file=../results/logs/high-jump-${dataset}.out
python3 ../nol/run_experiment.py -m high -p ${epsilon} -i $base_input -s $sample_para -o $output_folder -n 1 -iter $iterations -b $budget --reward $rewardfunction --save_gap $savegap --sampling-method $sampling_method --processes $processes --log $log_file 

## low 
output_folder=${base_output}/baseline-${rewardfunction}'-low/'
log_file=../results/logs/low-${dataset}.out
python3 ../nol/run_experiment.py -m  low -p ${epsilon} -i $base_input -s $sample_para -o $output_folder -n 1 -iter $iterations -b $budget --reward $rewardfunction --save_gap $savegap --sampling-method $sampling_method --processes $processes --log $log_file 


if [ $4 == 'wait' ]
then
	wait
fi

## KNN
#knn=${NOL}/baseline/net_complete/mab_explorer/
#cd ${knn}/mab_explorer/
#if [ ${dataset} == 'ba' ]
#then
#	knn_data='ba.txt'
#fi

#python sampling.py ${knn}/data/${knn_data} -s 0.01 -b $budget -e $iterations -m rn --results_dir ${knn}/results/
