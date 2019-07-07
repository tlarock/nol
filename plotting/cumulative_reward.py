#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:32:41 2019

@author: larock
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
argv = sys.argv

if len(argv) > 1:
    fig_num = argv[1]

title_size = 37
font_size = 30
tick_size = 32
label_size = 34
legend_font = 19

sample_dir = 'node-0.01/'

out_reward_name = 'new-nodes'

plots_base = '../results/plots/cumulative_reward/'
results_base = '../results/'
if len(argv) == 1 or fig_num == '3':
    fig_num = '3'
    start_probe = 0
    num_probes = 5000

    names = {
        'synthetic/ba-graph_N-10000_m-5_m0-5/':('BA',10000),
        'synthetic/N-10000_maxcc-0.95_maxgcc-0.15_avgDeg-10/': ('BTER', 10000),
        'cora/': ('Cora', 23000),
        'dblp/': ('DBLP', 6700),
        'enron/':('Enron', 36700),
        'caida/':('Caida', 26500)
    }
elif fig_num == '4':
    start_probe = 0
    num_probes = 5000

    names = {
        'synthetic/lfr-graph_N-34546_mu-0.1/': ('LFR-1', 34546),
        'synthetic/lfr-graph_N-34546_mu-0.2/': ('LFR-2', 34546),
        'synthetic/lfr-graph_N-34546_mu-0.3/': ('LFR-3', 34546),
        'synthetic/lfr-graph_N-34546_mu-0.4/': ('LFR-4', 34546)
    }
elif fig_num == '6':
    start_probe = 0
    num_probes = 50000

    names = {
        'twitter/': ('twitter', 90000)
    }
elif fig_num == '7':
    start_probe = 0
    num_probes = 1200
    out_reward_name = 'attribute'
    names = {
        'lj/':('lj', 1)
    }
elif fig_num == '8':
    start_probe = 0
    num_probes = 5000
    names = {
        'regular/':('Regular', 10000),
        'synthetic/er-graph_N-10000_p-0.001/':('ER', 10000)
    }
elif fig_num == '10':
    sample_dir = 'walk-0.01/'
    start_probe = 0
    num_probes = 5000

    names = {
        'synthetic/ba-graph_N-10000_m-5_m0-5/':('BA',10000),
        'synthetic/N-10000_maxcc-0.95_maxgcc-0.15_avgDeg-10/': ('BTER', 10000),
        'cora/': ('Cora', 23000),
        'dblp/': ('DBLP', 6700),
        'enron/':('Enron', 36700),
        'caida/':('Caida', 26500)
    }

else:
    print('Invalid figure number specified. Options are [3, 4, 6]. Nothing to do.')
    names = {}

for name in names:
    out_name = names[name][0]
    print(out_name)

    N = names[name][1]
    if out_name == 'livejournal' or out_name == 'patents' or out_name == 'lj':
        sample_dir = 'node-0.0001/'
    else:
        sample_dir = 'node-0.01/'

    if out_name == 'DBLP':
        num_probes = 4000
    elif out_name != 'DBLP' and fig_num == '3':
        num_probes = 5000

    input_dir = results_base + name + sample_dir

    if ((fig_num == '3' or fig_num == '9') and 'ba' in name) or (fig_num == '4' and out_name == 'LFR-1') or (fig_num == '7'):
        legend=True
    else:
        legend=False

    if fig_num != '7':
        nonpara_results = '/Users/larock/git/nol/baseline/net_complete/mab_explorer/results/' + names[name][0]+ '_rn_results'
        input_files = [
                (input_dir + 'default-new_nodes-NOL-epsilon-0.3-decay-1/network1/NOL_a0.01.csv', r'NOL($\epsilon=0.3$)', '-'),
                (input_dir + 'default-new_nodes-NOL-HTR-epsilon-0.3-decay-1/network1/NOL-HTR_a0.01.csv', r'NOL-HTR($\epsilon_0=0.3$,$k=\ln(n)$)', '-'),
                (nonpara_results + '_KNN-UCB.csv', r'KNN($k=20$)-UCB($\alpha=2.0$)', '-'),
                (input_dir + 'baseline-new_nodes-rand/network1/rand_a0.csv', 'Random', '-'),
                (input_dir + 'baseline-new_nodes-high-jump//network1/high_a0.csv', 'High + Jump', '-'),
                (input_dir + 'baseline-new_nodes-high/network1/high_a0.csv', 'High', '-'),
                (input_dir + 'baseline-new_nodes-low/network1/low_a0.csv', 'Low', '-'),
                ]
    else:
        sh_results = '/Users/larock/git/network_discovery/baselines/d3ts/src/mab/results/' + names[name][0] + '/dts.5_max_config2_all/extracted/' + names[name][0] + '.tsv'
        input_files = [
                (input_dir + 'netdisc-attribute-logit-epsilon-0.3-decay-1/network1/logit_a0.01.csv', r'NOL-BR($\epsilon=0.1$)', '-'),
                (sh_results, r'SelectiveHarvesting', '-')
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

        if N == 1:
            plt.xlabel('t')
        else:
            plt.xlabel('% Nodes Probed')
        plt.ylabel(r'Avg $c_r(t)$')
        if legend:
            leg = plt.legend()
            # set the linewidth of each legend object
            for legobj in leg.legendHandles:
                legobj.set_linewidth(3.0)

        if fig_num == '4':
            if '1' in out_name:
                plt.title(r'$\mu=0.1,Q=0.8$')
            elif '2' in out_name:
                plt.title(r'$\mu=0.2,Q=0.7$')
            elif '3' in out_name:
                plt.title(r'$\mu=0.3,Q=0.46$')
            elif '4' in out_name:
                plt.title(r'$\mu=0.4,Q=0.41$')


        plt.tight_layout()
        if names[name][0] == 'Twitter':
            inset_start_probe = 500
            inset_end_probe = 2000
            for i in range(len(input_files)):
                try:
                    df = pd.read_table(input_files[i][0])
                except Exception as e:
                    print(e)
                    df = None
                    continue

                a = plt.axes([.27, .697, .23, .23])
                a.tick_params(axis='x', labelsize=14)
                if 'default' not in input_files[i][0]:
                    a.plot(df['Probe'][inset_start_probe:inset_end_probe]/ float(N), df['AvgRewards'][inset_start_probe:inset_end_probe], color='C'+str(i+1), linewidth = 2.0, label=input_files[i][1], linestyle=input_files[i][2], alpha = 1.0)
                    a.fill_between(df['Probe'][inset_start_probe:inset_end_probe]/ float(N), yminus[inset_start_probe:inset_end_probe], yplus[inset_start_probe:inset_end_probe], color='C'+str(i+1), alpha=0.3)
                else:
                    a.plot(df['Probe'][inset_start_probe:inset_end_probe]/ float(N), df['AvgRewards'][inset_start_probe:inset_end_probe], linewidth = 2.0, label=input_files[i][1], linestyle=input_files[i][2], alpha = 1.0)
                    a.fill_between(df['Probe'][inset_start_probe:inset_end_probe]/ float(N), yminus[inset_start_probe:inset_end_probe], yplus[inset_start_probe:inset_end_probe], alpha=0.3)
                plt.setp(a, yticks=[])

        if fig_num != '9':
            plt.savefig(plots_base + '/' + str(out_name) + '-' + out_reward_name + '.pdf', dpi = 300)
        else:
            plt.savefig(plots_base + '/' + str(out_name) + '-' + out_reward_name + '-walk.pdf', dpi = 300)

