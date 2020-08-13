#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:50:15 2019

@author: larock
"""

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pickle

max_vals = {
    'lfr-0.1-2':(23471.6, 18066.6, 16783.9),
    'lfr-0.1-2.25':(25125.5, 17347.3, 20129.7),
    'lfr-0.1-2.5':(25852.8, 16402.2, 21074.6),
    'lfr-0.1-2.75':(25445.8, 15003.4, 21931.1),
    'lfr-0.1-3.0':(25354.6, 13885.3, 22739.0),
    'lfr-0.1-3.25':(24635.1, 11907.5, 22206.7),
    'lfr-0.1-3.5':(25078.4, 11842.6, 22733.0),
    'lfr-0.2-2':(25366.8, 22658.2, 17814.8),
    'lfr-0.2-2.25':(26854.2, 23133.1, 21486.1),
    'lfr-0.2-2.5':(27691.0, 23009.8, 22371.5),
    'lfr-0.2-2.75':(27682.8, 22654.5, 23455.1),
    'lfr-0.2-3.0':(27079.9, 20705.1, 23981.6),
    'lfr-0.2-3.25':(26290.7, 18767.2, 23489.4),
    'lfr-0.2-3.5':(26265.6, 17261.2, 24308.7),
    'lfr-0.3-2':(26652.0, 25075.4, 18082.3),
    'lfr-0.3-2.25':(28932.5, 27193.3, 22807.6),
    'lfr-0.3-2.5':(28629.3, 26389.8, 23264.9),
    'lfr-0.3-2.75':(28566.9, 26244.4, 23945.6),
    'lfr-0.3-3.0':(27968.6, 25737.1, 24749.3),
    'lfr-0.3-3.25':(26819.9, 23411.4, 24041.9),
    'lfr-0.3-3.5':(27221.4, 23055.0, 25023.4),
    'lfr-0.4-2':(28037.6, 27351.5, 18987.5),
    'lfr-0.4-2.25':(29091.9, 28162.5, 22893.8),
    'lfr-0.4-2.5':(29358.7, 28321.5, 23908.1),
    'lfr-0.4-2.75':(28888.2, 27949.8, 24404.7),
    'lfr-0.4-3.0':(28663.1, 27711.1, 25034.7),
    'lfr-0.4-3.25':(27723.6, 26508.1, 24477.6),
    'lfr-0.4-3.5':(27826.7, 26122.1, 25308.3),
    'lfr-0.5-2':(28355.1, 27987.3, 18911.2),
    'lfr-0.5-2.25':(29760.8, 29354.0, 23227.7),
    'lfr-0.5-2.5':(29809.3, 29612.5, 24246.4),
    'lfr-0.5-2.75':(29604.5, 29420.8, 24763.2),
    'lfr-0.5-3.0':(29334.9, 28956.3, 25279.3),
    'lfr-0.5-3.25':(28429.4, 27964.0, 24741.5),
    'lfr-0.5-3.5':(28396.7, 27820.8, 25437.0),
    'ba':(8998.2, 8994.95, 8915.45),
    'bter':(6604.4, 5379.8, 6235.9),
    'caida':(22181.2, 21557.5, 13622.35),
    'cora':(15743.6, 12384.95, 14051.9),
    'dblp':(4744.8, 3977.4, 4397.2),
    'enron':(27969.4, 25141.6, 17113.1),
    'twitter':(42237.7, 37541.1, 34998.7)
}

modularity_dict = pickle.load(open('../data/modularity_all.pickle', 'rb'))

degree_exp_dict = pickle.load(open('../data/degree_exponents_all.pickle', 'rb'))

x = []
y = []
c = []
for name in max_vals:
    x.append(degree_exp_dict[name])
    y.append(modularity_dict[name])
    c.append(((max_vals[name][0]-max_vals[name][1]) / max_vals[name][1] )*100)

print('min: {} max: {}'.format(min(c), max(c)))
plt.figure(figsize=(8, 6))
CMAP = get_cmap('inferno')
plt.title('HTR vs. High Degree')
plt.xlabel(r'Degree Exponent $\alpha$')
plt.ylabel(r'Modularity $Q$')
plt.scatter(x, y, cmap=CMAP, c=c)
plt.colorbar(label='% Performance Gain')
plt.savefig('../results/plots/HTR_vs_high.png', dpi=100)

for i, name in enumerate(max_vals):
    c[i] = ((max_vals[name][0]-max_vals[name][2]) / max_vals[name][2]) * 100

print('min: {} max: {}'.format(min(c), max(c)))
plt.figure(figsize=(8, 6))
plt.title('HTR vs. Random Degree')
plt.xlabel(r'Degree Exponent $\alpha$')
plt.ylabel(r'Modularity $Q$')
CMAP = get_cmap('inferno')
plt.scatter(x, y, cmap=CMAP, c=c)
plt.colorbar(label='% Performance Gain')
plt.savefig('../results/plots/HTR_vs_rand.png', dpi=100)