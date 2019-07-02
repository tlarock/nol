#!/bin/bash
set -ex

## Read config file ($1) passing parameters ($2=model, $3=feature type, $4=reward function)
source $1 $2 $3 $4 $5 $6

for p in $p_vals
do
echo $p
for a in $alpha_vals
do
	for sample_para in $sample_vals
	do
	for k in $k_vals
	do
			echo "START: " $output_folder
			output_folder=${base_output}${sample_para}/${featuretype}-${rewardfunction}'-'${model}'-epsilon-'${p}-decay-$decay/
			#python3 $program -m $model -i $base_input -s $sample_dir -o $output_folder -n $networks -iter $iterations -e $episodes -b $budget --alpha $a --feats $featuretype --reward $rewardfunction --save_gap $savegap -p $p --decay $decay --ktype $ktype -k $k --burn $burnin --sample $compute_samp --sampling-method $sampling_method &
			
			python3 $program -m $model --attr $attribute_file --target $target_attribute --seeds $seeds -i $base_input -s $sample_para -o $output_folder -n $networks -iter $iterations -b $budget --alpha $a --feats $featuretype --reward $rewardfunction --save_gap $savegap -p $p --decay $decay --ktype $ktype -k $k --burn $burnin --sampling-method $sampling_method --processes $processes &
			echo "END: " $output_folder
		    done
		done
done
done

## if you throw an extra parameter (whatever you want) to the script, it'll wait
#if [ $# -eq 7 ]
#then
wait
#fi
