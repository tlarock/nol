#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:18:32 2018

@author: larock
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
results_base = '../results/synthetic/'

inputs = [
    #(results_base + 'N-10000_maxcc-0.95_maxgcc-0.15_avgDeg-10', 'BTER'),
    (results_base + 'ba-graph_N-10000_m-5_m0-5', 'BA')
          ]

## initialize plot
font_size = 21
tick_size = 19
label_size = 20
legend_font = 19
plt.rcParams['xtick.labelsize'] = tick_size
plt.rcParams['ytick.labelsize'] = tick_size
plt.rcParams['axes.labelsize'] = label_size
plt.rcParams['font.size'] = font_size
plt.rcParams['legend.fontsize'] = legend_font

start_probe = 50
num_probes = 5000
#samples = ['0.01', '0.025', '0.05', '0.075', '0.1']
samples = ['0.01', '0.025']
network = 1
for generator, name in inputs: 
    data = defaultdict(list)
    plt.figure(figsize=(10,7))
    #plt.title(name + ' network')
    #plt.xlabel(r'$t$')
    plt.xlabel(r'% Nodes Probed')
    plt.ylabel(r'Avg $E(t)$')
    for iteration in range(0, 2):
        #plt.figure(figsize=(8,8))
        for sample in samples:
            values_list = []
            input_file = generator + '/node-' + sample + '/default-new_nodes-NOL-epsilon-0.3-decay-1/network' \
                        + str(network) + '/intermediate_results/NOL_iter' + str(iteration) + '_intermediate.txt'
            df = pd.read_table(input_file)
            #data[sample].append(df[0:num_probes].where(df['jump'] == 0)['delta'].fillna(0).abs().cumsum())
            data[sample].append(df[start_probe:num_probes]['delta'].fillna(0).abs().cumsum())
            min_index = np.where(df[start_probe:num_probes]['delta'] == df[start_probe:num_probes]['delta'].min())[0][0]
            #print(df[start_probe:num_probes]['delta'][0:min_index+1])
            #data[sample].append(df[0:num_probes]['delta'])
            #plt.plot(df[start_probe:num_probes]['delta'].fillna(0), label=str(sample))
        #plt.title(str(iteration))
        #plt.legend()
        #plt.savefig('../../results/plots/MoMs/PredictionError/BTER/iter-' + str(iteration) + '_error.pdf', dpi=150)
    #plt.figure(figsize=(10,7))
    for sample in sorted(data.keys()):
        y = np.array(data[sample])
        avgs = y.mean(axis = 0)
        stds = y.std(axis = 0)
        #plt.errorbar(range(len(avgs)), avgs, stds, label = str(100*float(sample)))
        x = np.linspace(start_probe/10000, num_probes/10000, avgs.shape[0])
        #plt.plot(range(start_probe, num_probes), avgs, label = str(100*float(sample)) + '%')
        plt.plot(x, avgs, label = str(100*float(sample)) + '%')
        #plt.yscale('log')
        plt.fill_between(x, avgs-stds, avgs+stds, alpha = 0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
    #plt.savefig('../../results/plots/MoMs/PredictionError/' + name + '_AveragePredictionError-50.png', dpi=200)
