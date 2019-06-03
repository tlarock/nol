#!/bin/bash

set -ex

dataset=$1

if [ ${dataset} == 'ba' ] 
then
	data_dir='synthetic/ba-graph_N-10000_m-5_m0-5/'
fi

if [ ${dataset} == 'bter' ]
then
	data_dir='synthetic/N-10000_maxcc-0.95_maxgcc-0.15_avgDeg-10/'
fi



base_input=${NOL}'/data/'${data_dir}
base_sample='node-sample-'
base_output=${NOL}'/results/'${data_dir}${base_sample}
sample_para=0.01
sample_dir=${base_sample}${sample_para}
iterations='10'
budget='5000'
alpha=0.01
featuretype='default'
rewardfunction='new_nodes'
savegap=0
epsilon=0.3
decay=0
burnin=0
compute_samp=False
processes=5

## NOL(\epsilon=0.3)
output_folder=${base_output}${sample_para}/${featuretype}-${rewardfunction}'-NOL-epsilon-'${epsilon}-decay-$decay/

python3 ../nol/run_experiment.py -m globalmax_restart -i $base_input -s $sample_dir -o $output_folder -n 1 -iter $iterations -b $budget --alpha $alpha --feats $featuretype --reward $rewardfunction --save_gap $savegap -p $epsilon --decay $decay --burn $burnin --sample $compute_samp --processes $processes &


## NOL-HTR(\epsilon=0.3, k=ln(n))
ktype=funct
k=np.log
output_folder=${base_output}${sample_para}/${featuretype}-${rewardfunction}'-NOL-HTR-epsilon-'${epsilon}-decay-$decay/

python3 ../nol/run_experiment.py -m NOL-HTR -i $base_input -s $sample_dir -o $output_folder -n 1 -iter $iterations -b $budget --alpha $alpha --feats $featuretype --reward $rewardfunction --save_gap $savegap -p $epsilon --decay $decay --ktype $ktype -k $k --burn $burnin --sample $compute_samp --processes $processes &

## Random
output_folder=${base_output}${sample_para}/${featuretype}-${rewardfunction}'-rand/'

python3 ../nol/run_experiment.py -m rand -i $base_input -s $sample_dir -o $output_folder -n 1 -iter $iterations -b $budget --reward $rewardfunction --save_gap $savegap --sample $compute_samp --processes $processes &

## High
output_folder=${base_output}${sample_para}/${featuretype}-${rewardfunction}'-high/'

python3 ../nol/run_experiment.py -m high -i $base_input -s $sample_dir -o $output_folder -n 1 -iter $iterations -b $budget --reward $rewardfunction --save_gap $savegap --sample $compute_samp --processes $processes &

## High-jump
output_folder=${base_output}${sample_para}/${featuretype}-${rewardfunction}'-high-jump/'

python3 ../nol/run_experiment.py -m high -p ${epsilon} -i $base_input -s $sample_dir -o $output_folder -n 1 -iter $iterations -b $budget --reward $rewardfunction --save_gap $savegap --sample $compute_samp --processes $processes &

wait
## KNN
#knn=${NOL}/baseline/net_complete/mab_explorer/
#cd ${knn}/mab_explorer/
#if [ ${dataset} == 'ba' ]
#then
#	knn_data='ba.txt'
#fi

#python sampling.py ${knn}/data/${knn_data} -s 0.01 -b $budget -e $iterations -m rn --results_dir ${knn}/results/
