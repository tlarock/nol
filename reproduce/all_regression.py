#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:42:18 2019

@author: larock
"""
#from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.formula.api as smf
from itertools import product
from leiden_modularity import get_leiden_modularity
#from statsmodels.discrete.discrete_model import MNLogit
import numpy as np
import networkx as nx
import community
import powerlaw
import pickle
#import matplotlib.pyplot as plt
#import warnings
#warnings.filterwarnings(action='ignore', category=DeprecationWarning)

data_dir = '../data/'


data_base = '../data/synthetic/lfr/'
path_template = data_base + 'lfr-graph-mu-{}-alpha-{}/network1/network1.txt'
#default-new_nodes-NOL-HTR-k-{}-epsilon-{}-decay-{}/network1/NOL-HTR_a0.01.csv'
## specify values of k and p
k_vals = list([0, 1, 2, 4, 8] + ['np.log2', 'np.log', 'np.log10'] + [16, 32, 64, 128])
epsilon_vals = [0, 0.1, 0.2, 0.3, 0.4]
htr_iter = list(product(k_vals, epsilon_vals))

## Specify mu and alpha values
mu_vals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
alpha_vals = [2, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]
lfr_iter = list(product(mu_vals, alpha_vals))

network_paths = {'lfr-{}-{}'.format(mu, alpha): path_template.format(mu,alpha) \
                 for mu, alpha in lfr_iter}
lfr_only = False
leiden = True

## Add real networks and BTER/BA to the list of networks to consider
if not lfr_only:
    for real_net in ['caida', 'twitter', 'dblp', 'enron', 'cora']:
        network_paths[real_net] = data_dir + real_net + '/network1/network1.txt'
    network_paths['ba'] = data_dir + 'synthetic/ba-graph_N-10000_m-5_m0-5/network1/network1.txt'
    network_paths['bter'] = data_dir + 'synthetic/N-10000_maxcc-0.95_maxgcc-0.15_avgDeg-10/network1/network1.txt'

# k_maps = {'ln':200, 'log':201, 'log2':202}
best_parameters = {
        'lfr-0-2':(128, 0, 0),
        'lfr-0-2.25':(32, 0, 0),
        'lfr-0-2.5':(32, 0, 0),
        'lfr-0-2.75':('log2', 0, 0),
        'lfr-0-3.0':(2, 0, 0),
        'lfr-0-3.25':(8, 0, 0),
        'lfr-0-3.5':('log', 0, 0),
        'lfr-0.1-2':(1, 0, 0),
        'lfr-0.1-2.25':(8, 0.4, 0),
        'lfr-0.1-2.5':(16, 0.4, 0),
        'lfr-0.1-2.75':(16, 0.4, 0),
        'lfr-0.1-3.0':(128, 0.4, 0),
        'lfr-0.1-3.25':(128, 0.4, 0),
        'lfr-0.1-3.5':(128, 0.4, 0),
        'lfr-0.2-2':(1, 0.4, 0),
        'lfr-0.2-2.25':('log', 0.4, 0),
        'lfr-0.2-2.5':('log10', 0, 0),
        'lfr-0.2-2.75':(1, 0, 0),
        'lfr-0.2-3.0':('log2', 0.4, 0),
        'lfr-0.2-3.25':(128, 0.4, 0),
        'lfr-0.2-3.5':(128, 0.4, 0),
        'lfr-0.3-2':(8, 0.3, 0),
        'lfr-0.3-2.25':('log', 0, 0),
        'lfr-0.3-2.5':('log10', 0, 0),
        'lfr-0.3-2.75':(1, 0, 0),
        'lfr-0.3-3.0':('log', 0.4, 0),
        'lfr-0.3-3.25':(128, 0.4, 0),
        'lfr-0.3-3.5':(128, 0.4, 0),
        'lfr-0.4-2':('log2', 0, 0),
        'lfr-0.4-2.25':(2, 0, 0),
        'lfr-0.4-2.5':(1, 0, 0),
        'lfr-0.4-2.75':(1, 0, 0),
        'lfr-0.4-3.0':('log', 0.2, 0),
        'lfr-0.4-3.25':(1, 0.4, 0),
        'lfr-0.4-3.5':(1, 0.4, 0),
        'lfr-0.5-2':(2, 0, 0),
        'lfr-0.5-2.25':(2, 0, 0),
        'lfr-0.5-2.5':('log10', 0, 0),
        'lfr-0.5-2.75':(1, 0, 0),
        'lfr-0.5-3.0':(1, 0, 0),
        'lfr-0.5-3.25':(1, 0.2, 0),
        'lfr-0.5-3.5':('log10', 0.3, 0),
        'lfr-0.6-2':(128, 0, 0),
        'lfr-0.6-2.25':(32, 0, 0),
        'lfr-0.6-2.5':(16, 0, 0),
        'lfr-0.6-2.75':('log10', 0, 0),
        'lfr-0.6-3.0':('log10', 0, 0),
        'lfr-0.6-3.25':(2, 0, 0),
        'lfr-0.6-3.5':(2, 0, 0),
        'lfr-0.7-2':(128, 0.1, 0),
        'lfr-0.7-2.25':(64, 0.1, 0),
        'lfr-0.7-2.5':(32, 0, 0),
        'lfr-0.7-2.75':('log2', 0, 0),
        'lfr-0.7-3.0':(2, 0, 0),
        'lfr-0.7-3.25':(8, 0, 0),
        'lfr-0.7-3.5':(2, 0, 0),
}

if not lfr_only:
    best_parameters['ba'] = (32, 0.0, 0)
    best_parameters['bter'] = (128, 0.4, 0)
    best_parameters['caida'] = (16, 0.2, 1)
    best_parameters['cora'] = (1, 0.4, 0)
    best_parameters['dblp'] = (32, 0.0, 0)
    best_parameters['enron'] = (3, 0.0, 0)
    best_parameters['twitter'] = (2, 0.2, 0)
   
# store true values
y = [[], [], []]
idx = 0
for name in best_parameters:
    if name in network_paths:
        y[0].append(str(best_parameters[name][0]))
        y[1].append(best_parameters[name][1])
        y[2].append(best_parameters[name][2])
  
X = np.zeros((len(network_paths), 4))

transitivity_dict = dict()
modularity_dict = dict()
degree_exp_dict = dict()

## Hard coding some feature computations so I don't 
## have to keep computing them  
 
try:
    transitivity_dict = pickle.load(open('../data/transitivity_dict.pickle', 'rb'))
except Exception as e:
    print(e)

try:
    if leiden:
        modularity_dict = pickle.load(open('../data/modularity_leiden.pickle', 'rb'))
    else:
        modularity_dict = pickle.load(open('../data/modularity_louvain.pickle', 'rb'))
except Exception as e:
   print(e)  
    
try:
    degree_exp_dict = pickle.load(open('../data/degree_exp_dict.pickle', 'rb'))
except Exception as e:
    print(e)

idx = 0
for name, network_path in network_paths.items():
    #print('Network: ' + name)
    G = nx.read_adjlist(network_path)
    
    if name in transitivity_dict:
        transitivity = transitivity_dict[name]
    else:
        transitivity = nx.transitivity(G)
        transitivity_dict[name] = transitivity

    X[idx][0] = transitivity
    
    if name in modularity_dict:
        modularity = modularity_dict[name]
    else:
        if leiden:
            modularity = get_leiden_modularity(G)
        else:
            partition = community.best_partition(G)
            modularity = community.modularity(partition, G)
        
        modularity_dict[name] = modularity
        
    X[idx][1] = modularity
    
    if name in degree_exp_dict:
        degree_exponent = degree_exp_dict[name]
    else:
        degree_sequence = [d for n, d in G.degree()]
        pl = powerlaw.Fit(degree_sequence, verbose=False)
        degree_exponent = pl.power_law.alpha
        degree_exp_dict[name] = degree_exponent

    X[idx][2] = degree_exponent
    
    #print("transitivity: {}, alpha: {}".format(\
    #      transitivity, degree_exponent))
    idx+=1


import pandas as pd

data = pd.DataFrame({'epsilon': y[1], 'k': y[0], 'clustering':X[:, 0], \
                     'modularity':X[:,1], 'alpha':X[:,2] }, \
                     index=network_paths.keys())

## ToDo I don't think doing this is really necessary anymore
## I already have transitivity_dict filled up?
transitivity_dict = dict(data.clustering)
try:
    outfile = open('../data/transitivity_dict.pickle', 'wb')
    pickle.dump(transitivity_dict, outfile)
    outfile.close()
except Exception as e:
    print(e)
 
modularity_dict = dict(data.modularity)
try:
    if leiden:
        outfile = open('../data/modularity_leiden.pickle', 'wb')
    else:
        outfile = open('../data/modularity_louvain.pickle', 'wb')
    pickle.dump(modularity_dict, outfile)
    outfile.close()
except Exception as e:
    print(e)
    

degree_exp_dict = dict(data.alpha)
try:
    outfile = open('../data/degree_exp_dict.pickle', 'wb')
    pickle.dump(degree_exp_dict, outfile)
    outfile.close()
except Exception as e:
    print(e)

######################## LINEAR REGRESSION ON EPSILON ########################
model = smf.ols(formula='epsilon ~ clustering + modularity + alpha', data=data)
results = model.fit()
print(results.summary())
