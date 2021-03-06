#!/bin/bash
#set -ex

program='../nol/run_experiment.py'
networks=1
burnin=0
model='NOL-HTR'
iterations='10'
budget='5000'
savegap='0'
featuretype='default'
rewardfunction='new_nodes'
p_vals='0 0.1 0.2 0.3 0.4'
k_vals="1 2 4 8 16 32 64 128"
ktype='int'
decay_vals='0 1'
sample_para=0.01
a=0.01
sampling_method='node'
base_input='../data/synthetic/lfr/lfr-graph_N-34546-mu-${mu}-${alpha}/'
base_output='../results/synthetic/lfr/lfr-graph_N-34546-mu-${mu}-alpha-${alpha}/'$sampling_method-
processes=1


for decay in $decay_vals
do
    for p in $p_vals
    do
        for k in $k_vals
        do
            output_folder=${base_output}${sample_para}/${featuretype}-${rewardfunction}-${model}-k-${k}-epsilon-${p}-decay-$decay/
            python3 $program -m $model -i $base_input -s $sample_para -o $output_folder -n $networks -iter $iterations -b $budget --alpha $a --feats $featuretype --reward $rewardfunction --save_gap $savegap -p $p --decay $decay --ktype $ktype -k $k --burn $burnin --sampling-method $sampling_method --processes $processes &
         done
    done
done


ktype='funct'
k_vals="np.log np.log10 np.log2"
for decay in $decay_vals
do
    for p in $p_vals
    do
        for k in $k_vals
        do
            output_folder=${base_output}${sample_para}/${featuretype}-${rewardfunction}-${model}-k-${k}-epsilon-${p}-decay-$decay/
            python3 $program -m $model -i $base_input -s $sample_para -o $output_folder -n $networks -iter $iterations -b $budget --alpha $a --feats $featuretype --reward $rewardfunction --save_gap $savegap -p $p --decay $decay --ktype $ktype -k $k --burn $burnin --sampling-method $sampling_method --processes $processes &
         done
    done
done

wait
