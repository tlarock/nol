#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:53:38 2019

This script finds the best performing (in terms of final cumulative reward) parameters from
the LFR parameter search and prints them out in a dictionary format that can be copied into
a file to be read or an interpreter.

@author: larock
"""
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict

output=False
plot=False

if output:
    font_size = 55
    tick_size = 45
    label_size = 50
    fig_size = (14,14)
else:
    font_size = 34
    tick_size = 24
    label_size = 24
    fig_size = (9,9)
    
 
results_base = '../results/synthetic/lfr/'
path_template = results_base + 'lfr-graph-mu-{}-alpha-{}/node-0.01/default-new_nodes-NOL-HTR-k-{}-epsilon-{}-decay-{}/network1/NOL-HTR_a0.01.csv'
high_template = results_base + 'lfr-graph-mu-{}-alpha-{}/node-0.01/baseline-new_nodes-high/network1/high_a0.csv'
rand_template = results_base + 'lfr-graph-mu-{}-alpha-{}/node-0.01/baseline-new_nodes-rand/network1/rand_a0.csv'
## specify values of k and p
k_vals = list([1, 2, 4, 8] + ['np.log2', 'np.log', 'np.log10'] + [16, 32, 64, 128])
epsilon_vals = [0, 0.1, 0.2, 0.3, 0.4]
htr_iter = list(product(k_vals, epsilon_vals))

## Specify mu and alpha values
mu_vals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
alpha_vals = [2, 2.25, 2.5, 2.75, 3., 3.25, 3.5]
lfr_iter = list(product(mu_vals, alpha_vals))

decay = 0

results_dict = dict()
print('{')
for mu, alpha in lfr_iter:
    for k, epsilon in htr_iter:
        filepath = path_template.format(mu, alpha, k, epsilon, decay)
        try:
            data = pd.read_table(filepath)
        except Exception:
            print("No file at {}".format(filepath))
            continue

        ## normalize by the max average reward to 
        ## compare across networks
        val = data['AvgRewards'].max()
        #val = data['AvgRewards'][int(0.5*data.AvgRewards.shape[0])]
        results_dict.setdefault((mu, alpha), dict())
        results_dict[(mu,alpha)][(k, epsilon, decay)] = val
        #print('(mu: {}, alpha: {}), (k: {}, epsilon: {}): {}'.format(mu, alpha, \
              #k, epsilon, results_dict[(mu, alpha)][(k, epsilon, decay)]))
    
    try:
        (max_k, max_eps, max_decay), max_val = max(results_dict[(mu, alpha)].items(), key=lambda kv: kv[1])
        if isinstance(max_k, str):
            max_k = "\'{}\'".format(max_k.split('np.')[1])

        #print('Best Performing Parameters for (mu={}, alpha={}): k={}, epsilon={}'.format(mu, alpha, max_k, max_eps))
        ## This line outputs the best performing parameters
        print("\t\'lfr-{}-{}\':({}, {}, {}),".format(mu, alpha, max_k, max_eps, max_decay))
        
        ## these lines output the maximums for the best HTR, high and random
        #high_max = pd.read_table(high_template.format(mu, alpha)).AvgRewards.max()
        #rand_max = pd.read_table(rand_template.format(mu, alpha)).AvgRewards.max()
        #high_max = pd.read_table(high_template.format(mu, alpha)).AvgRewards[2500]
        #rand_max = pd.read_table(rand_template.format(mu, alpha)).AvgRewards[2500]
        #print("\t\'lfr-{}-{}\':({}, {}, {}),".format(mu, alpha, max_val, high_max, rand_max))
    except Exception as e:
        print(e)
        
    
print('}')
