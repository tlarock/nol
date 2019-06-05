#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:32:41 2019

@author: larock
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

legend=True
title_size = 37
font_size = 30
tick_size = 32
label_size = 34
legend_font = 19

N = 10000
start_probe = 0
num_probes = 5000

sample_dir = 'node-sample-0.01/'
plots_base = '../results/plots/cumulative_reward/'
results_base = '../results/synthetic/'
name = 'ba-graph_N-10000_m-5_m0-5/'
name = 'N-10000_maxcc-0.95_maxgcc-0.15_avgDeg-10/'

names = {
    'ba-graph_N-10000_m-5_m0-5/':'BA',
    'N-10000_maxcc-0.95_maxgcc-0.15_avgDeg-10/':'BTER'
}

for name in names:
    input_dir = results_base + name + sample_dir

    nonpara_results = '/Users/larock/git/nol/baseline/net_complete/mab_explorer/results/ba_rn_results'
    input_files = [
            (input_dir + 'default-new_nodes-NOL-epsilon-0.3-decay-1/network1/.csv', r'NOL($\epsilon=0.3$)', '-'),
            (input_dir + 'default-new_nodes-NOL-HTR-epsilon-0.3-decay-1/network1/NOL-HTR_a0.01.csv', r'NOL-HTR($\epsilon_0=0.3$,$k=\ln(n)$)', '-'),
            #(nonpara_results + '_KNN-UCB.csv', r'KNN($k=20$)-UCB($\alpha=2.0$)', '-'),
            (input_dir + 'baseline-new_nodes-rand/network1/rand_a0.csv', 'Random', '-'),
            (input_dir + 'baseline-new_nodes-high-jump//network1/high_a0.csv', 'High + Jump', '-'),
            (input_dir + 'baseline-new_nodes-high/network1/high_a0.csv', 'High', '-'),
            ]


    ## initialize plot
    plt.figure(figsize=(10, 8))
    plt.rcParams['xtick.labelsize'] = tick_size
    plt.rcParams['ytick.labelsize'] = tick_size
    plt.rcParams['axes.labelsize'] = label_size
    plt.rcParams['font.size'] = font_size
    plt.rcParams['legend.fontsize'] = legend_font
    for i in range(len(input_files)):
        try:
            df = pd.read_table(input_files[i][0])
        except Exception as e:
            print(e)
            df = None
            continue
        avg = np.array(df['AvgRewards'][start_probe:num_probes])
        std = np.array(df['StdRewards'][start_probe:num_probes])
        yminus = avg-std
        yplus = avg+std

        if 'default' not in input_files[i][0]:
            plt.plot(df['Probe'][start_probe:num_probes]/ float(N), df['AvgRewards'][start_probe:num_probes], color = 'C'+str(i+1), linewidth = 2.0, label=input_files[i][1], linestyle=input_files[i][2], alpha = 1.0)
            plt.fill_between(df['Probe'][start_probe:num_probes]/ float(N), yminus, yplus, color = 'C'+str(i+1), alpha=0.3)
        else:
            plt.plot(df['Probe'][start_probe:num_probes]/ float(N), df['AvgRewards'][start_probe:num_probes], linewidth = 2.0, label=input_files[i][1], linestyle=input_files[i][2], alpha = 1.0)
            plt.fill_between(df['Probe'][start_probe:num_probes]/ float(N), yminus, yplus, alpha=0.3)
        print('label ' + str(input_files[i][1]) + ' max: ' + str(max(df['AvgRewards'][start_probe:num_probes])))

    if df is not None:
        title = str(name) + ' network'
        out_reward_name = 'new-nodes'

        if N == 1:
            plt.xlabel('t')
        else:
            plt.xlabel('% Nodes Probed')
        plt.ylabel(r'Avg $c_r(t)$')
        #plt.ylabel('Average Cumulative Reward')
        #leg = plt.legend(frameon=False)
        if legend:
            leg = plt.legend()
            # set the linewidth of each legend object
            for legobj in leg.legendHandles:
                legobj.set_linewidth(3.0)

        plt.tight_layout()
        out_name = names[name]
        plt.savefig(plots_base + '/' + str(out_name) + '-' + out_reward_name + '-HTR-inset.pdf', dpi = 300)
