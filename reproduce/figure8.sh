#!/bin/bash
# NOTE: This script relies on the NOL enviornment variable pointing to
# the highest level directory. e.g. export NOL=path/to/nol
set -ex

processes=$1

dataset='lj'
data_dir='lj/'
attribute_file='../data/lj/network1/attributes.txt'
target_attribute=0
seeds=2
iterations='10'
budget='500'
alpha=0.01
featuretype='netdisc'
rewardfunction='attribute'
savegap=0
burnin=20
sampling_method='netdisc'
base_input=${NOL}'/data/'${data_dir}
base_output=${NOL}'/results/'${data_dir}${sampling_method}

epsilon=0.0
decay=0
output_folder=${base_output}/${featuretype}-${rewardfunction}'-mod-epsilon-'${epsilon}-decay-$decay/
python3 ../nol/run_experiment.py -m mod --attr $attribute_file --target $target_attribute --seeds $seeds -i $base_input -o $output_folder -iter $iterations -b $budget --alpha $alpha --feats $featuretype --reward $rewardfunction --save_gap $savegap -p $epsilon --decay $decay --burn $burnin --sampling-method $sampling_method --processes $processes &

epsilon=0.3
decay=1
output_folder=${base_output}/${featuretype}-${rewardfunction}'-logit-epsilon-'${epsilon}-decay-$decay/
python3 ../nol/run_experiment.py -m logit --attr $attribute_file --target $target_attribute --seeds $seeds -i $base_input -o $output_folder -iter $iterations -b $budget --alpha $alpha --feats $featuretype --reward $rewardfunction --save_gap $savegap -p $epsilon --decay $decay --burn $burnin --sampling-method $sampling_method --processes $processes &

epsilon=0.1
decay=1
output_folder=${base_output}/${featuretype}-${rewardfunction}'-logit-epsilon-'${epsilon}-decay-$decay/
python3 ../nol/run_experiment.py -m logit --attr $attribute_file --target $target_attribute --seeds $seeds -i $base_input -o $output_folder -iter $iterations -b $budget --alpha $alpha --feats $featuretype --reward $rewardfunction --save_gap $savegap -p $epsilon --decay $decay --burn $burnin --sampling-method $sampling_method --processes $processes

wait

## PLOTTING CODE
#cd ../plotting/
#python cumulative_reward.py 9