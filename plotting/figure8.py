#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#one liner to verify computing the correct averages: 
#    awk -F "\t" 'BEGIN{avg = 0; total = -1}{if(total > -1){avg += $6}; total += 1} END{avg = avg / total; print avg}' LTD_globalmax_iter0_a0.01_l0.0_g0.0_episode_0_intermediate.txt 
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
from collections import defaultdict
import numpy as np

def getDataPerIteration(input_dir, max_iter = float('inf'), start_probe = 0, end_probe = 5000, verbose = False):
    files = sorted(glob.glob(input_dir + '/*'))
    values_list = defaultdict(list)
    ## loop through all of the intermediate files
    for file in files:
        iterName = re.findall('NOL_iter[0-9]+', file)
        if verbose:
            print(iterName)
        if len(iterName) > 0:
            iteration = iterName[0].strip().split('iter')[1]
            if verbose:
                print('iter: ' + str(iteration))
            ## read the data
            df = pd.read_table(file)
        else:
            continue
        ## Not sorting the file names, a bit quirky but will work..
        if int(iteration) < max_iter:
            if verbose:
                print(iteration)
            for i in range(0,5):
                if verbose:
                    print('Theta[' + str(i) + '] mean: ' + str(df['Theta[' + str(i) + ']'].mean()))
                values_list[iteration].append(df['Theta[' + str(i) + ']'][start_probe:end_probe])

    print('number of iterations: ' + str(len(values_list)))
    return values_list

font_size = 30
tick_size = 32
label_size = 34
legend_font = 19


base = '../results/synthetic/'
suffix = '/node-0.01/default-new_nodes-NOL-epsilon-0.3-decay-1/network1/intermediate_results/'
input_dirs = [
                (base + 'N-10000_maxcc-0.95_maxgcc-0.15_avgDeg-10' + suffix, 'BTER'),
                (base + 'ba-graph_N-10000_m-5_m0-5' + suffix, 'BA'),
              ]


thetaMapping = {0:'degree', 1:'clustering', 2:'connected component', 3:'fraction probed neighbors', 4:'lost reward'}
#thetaMapping = {1:'clustering', 2:'connected component', 3:'fraction probed neighbors', 4:'lost reward'}
max_iter = float('inf')
#max_iter = 1
#plot_type=''
plot_type = 'avg'

start_probe = 50
end_probe = 5000

colors = ['blue', 'orange', 'green', 'red', 'purple']

for input_dir, model_name in input_dirs:
    print('generator: ' + str(model_name))
    values = getDataPerIteration(input_dir, max_iter = max_iter, start_probe=start_probe, end_probe=end_probe, verbose=False)
    plt.figure(figsize=(10, 8))
    plt.rcParams['xtick.labelsize'] = tick_size
    plt.rcParams['ytick.labelsize'] = tick_size
    plt.rcParams['axes.labelsize'] = label_size
    plt.rcParams['font.size'] = font_size
    plt.rcParams['legend.fontsize'] = legend_font

    plt.title(model_name)
    data = defaultdict(list)
    ## for each iteration
    for iteration in sorted(values.keys()):
        ## for each theta
        for i in range(len(values[iteration])):
            y = values[iteration][i]
            if plot_type == 'avg':
                ## Collect the data to be averaged
                data[y.name].append(y)
            else:
                ## plot a curve per iteration
                if iteration == '0':
                    plt.plot(range(start_probe, end_probe), y, color = colors[i], label = thetaMapping[i])
                else:
                    plt.plot(range(start_probe, end_probe), y, color = colors[i], label = '_nolegend_')

    if plot_type == 'avg':
        ## plot averaged data
        for i in range(len(values[iteration])):
            y = values[iteration][i]
            label = y.name
            avg = np.mean(np.array(data[label]), axis = 0)
            std = np.std(np.array(data[label]), axis = 0)
            x_max = min(avg.shape[0]+start_probe, end_probe)
            plt.plot(range(start_probe, x_max), avg, color = colors[i], label = thetaMapping[i])
            plt.fill_between(range(start_probe, x_max), avg-std, avg+std, color = colors[i], alpha = 0.3)

    plt.ylabel('weight')
    plt.xlabel('probe')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../results/plots/feature_weights/' + model_name + '-features.pdf', dpi = 300)
