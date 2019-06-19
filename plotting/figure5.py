#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:18:32 2018

@author: larock
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
results_base = '../results/synthetic/'

inputs = [
    (results_base + 'N-10000_maxcc-0.95_maxgcc-0.15_avgDeg-10', 'BTER'),
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
samples = ['0.01', '0.025', '0.05', '0.075', '0.1']
network = 1
for generator, name in inputs:
    data = defaultdict(list)
    plt.figure(figsize=(10,7))
    plt.xlabel(r'% Nodes Probed')
    plt.ylabel(r'Avg $E(t)$')
    for iteration in range(0, 1):
        for sample in samples:
            values_list = []
            input_file = generator + '/node-' + sample + '/default-new_nodes-NOL-HTR-epsilon-0.3-decay-1/network' \
                        + str(network) + '/intermediate_results/NOL-HTR_iter' + str(iteration) + '_intermediate.txt'
            df = pd.read_table(input_file)
            data[sample].append(df[start_probe:num_probes]['delta'].fillna(0).abs().cumsum())

    for sample in sorted(data.keys()):
        y = np.array(data[sample])
        avgs = y.mean(axis = 0)
        stds = y.std(axis = 0)
        x = np.linspace(start_probe/10000, num_probes/10000, avgs.shape[0])
        plt.plot(x, avgs, label = str(100*float(sample)) + '%')
        plt.fill_between(x, avgs-stds, avgs+stds, alpha = 0.4)

    plt.legend()
    plt.tight_layout()
    plt.savefig('../results/plots/error/' + name + '_AveragePredictionError-50.png', dpi=200)
