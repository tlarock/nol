#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:12:10 2019

@author: larock
"""
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pickle
import cmocean


font_size = 25
tick_size = 15
label_size = 20
plt.rcParams['xtick.labelsize'] = tick_size
plt.rcParams['ytick.labelsize'] = tick_size
plt.rcParams['axes.labelsize'] = label_size
plt.rcParams['font.size'] = font_size

max_vals = {
        'lfr-0-2':(27376.7, 28091.7, 14194.7),
        'lfr-0-2.25':(29222.3, 29614.1, 18035.9),
        'lfr-0-2.5':(28959.1, 29290.6, 18485.1),
        'lfr-0-2.75':(28491.4, 28952.1, 18470.7),
        'lfr-0-3.0':(28099.4, 28463.9, 18845.6),
        'lfr-0-3.25':(26141.0, 26244.9, 17743.1),
        'lfr-0-3.5':(25615.3, 25720.6, 18692.6),
        'lfr-0.1-2':(20294.7, 14988.9, 12309.2),
        'lfr-0.1-2.25':(21220.4, 14046.3, 14968.1),
        'lfr-0.1-2.5':(21781.9, 12959.3, 15477.7),
        'lfr-0.1-2.75':(21177.0, 11102.0, 16049.4),
        'lfr-0.1-3.0':(20436.2, 10000.2, 16544.9),
        'lfr-0.1-3.25':(18568.7, 8020.9, 15725.4),
        'lfr-0.1-3.5':(18061.8, 7818.5, 16097.1),
        'lfr-0.2-2':(23289.5, 19900.1, 12987.7),
        'lfr-0.2-2.25':(24145.2, 19733.7, 16226.7),
        'lfr-0.2-2.5':(24381.6, 19461.1, 16508.8),
        'lfr-0.2-2.75':(24901.6, 19217.7, 17387.5),
        'lfr-0.2-3.0':(23167.7, 15930.7, 17505.7),
        'lfr-0.2-3.25':(20513.5, 13072.8, 16623.8),
        'lfr-0.2-3.5':(19940.5, 11498.0, 17343.4),
        'lfr-0.3-2':(24915.9, 22836.7, 13176.1),
        'lfr-0.3-2.25':(27235.1, 24615.4, 17388.1),
        'lfr-0.3-2.5':(26299.7, 23189.7, 17524.7),
        'lfr-0.3-2.75':(26129.4, 22830.1, 17569.4),
        'lfr-0.3-3.0':(24613.1, 21699.9, 18131.2),
        'lfr-0.3-3.25':(21965.0, 18577.7, 17225.5),
        'lfr-0.3-3.5':(21803.8, 17506.2, 17950.2),
        'lfr-0.4-2':(26234.5, 25452.4, 14180.7),
        'lfr-0.4-2.25':(27207.1, 26080.8, 17419.0),
        'lfr-0.4-2.5':(27381.6, 25699.6, 17885.1),
        'lfr-0.4-2.75':(26429.7, 25248.7, 18092.5),
        'lfr-0.4-3.0':(25922.3, 24620.0, 18521.4),
        'lfr-0.4-3.25':(23648.6, 22254.5, 17489.8),
        'lfr-0.4-3.5':(23019.8, 20667.9, 18239.5),
        'lfr-0.5-2':(26327.1, 26250.6, 14031.1),
        'lfr-0.5-2.25':(28014.0, 27692.0, 17669.9),
        'lfr-0.5-2.5':(28115.9, 27974.5, 18489.2),
        'lfr-0.5-2.75':(27662.6, 27339.6, 18445.8),
        'lfr-0.5-3.0':(26818.9, 26349.2, 18678.7),
        'lfr-0.5-3.25':(24872.6, 24086.0, 17774.7),
        'lfr-0.5-3.5':(24216.4, 23369.2, 18356.4),
        'lfr-0.6-2':(29044.2, 29198.0, 15038.7),
        'lfr-0.6-2.25':(30007.1, 30082.7, 18138.1),
        'lfr-0.6-2.5':(29385.2, 29517.7, 18458.2),
        'lfr-0.6-2.75':(28760.1, 28925.2, 18749.1),
        'lfr-0.6-3.0':(28213.4, 28430.9, 19006.2),
        'lfr-0.6-3.25':(25489.6, 25604.8, 17813.5),
        'lfr-0.6-3.5':(25485.0, 25641.1, 18772.9),
        'lfr-0.7-2':(28802.0, 29068.7, 14856.7),
        'lfr-0.7-2.25':(29704.1, 29931.6, 18010.7),
        'lfr-0.7-2.5':(29208.6, 29530.5, 18659.0),
        'lfr-0.7-2.75':(28840.9, 29071.3, 18767.7),
        'lfr-0.7-3.0':(28214.2, 28460.6, 19050.7),
        'lfr-0.7-3.25':(25871.1, 25993.2, 17843.9),
        'lfr-0.7-3.5':(25806.9, 25897.5, 18773.6),
        
        'ba':(8827.9, 8838.6, 7654.85),
        'bter':(6604.4, 5379.8, 6235.9),
        'caida':(22181.2, 21557.5, 13622.35),
        'cora':(15743.6, 12384.95, 14051.9),
        'dblp':(4048.4, 3106.0, 3420.4),
        'enron':(27969.4, 25141.6, 17113.1),
        'twitter':(42237.7, 37541.7, 35000.05),
}

modularity_dict = pickle.load(open('../data/modularity_leiden.pickle', 'rb'))


degree_exp_dict = pickle.load(open('../data/degree_exp_dict.pickle', 'rb'))

real_nets = ['ba', 'bter', 'caida', 'cora', 'dblp', 'enron', 'twitter']
x = []
y = []
c_high = []
c_rand = []
edgecolor = []
for name in max_vals:
    x.append(-degree_exp_dict[name])
    y.append(modularity_dict[name])
    c_high.append(((max_vals[name][0]-max_vals[name][1]) / max_vals[name][1] )*100)
    c_rand.append(((max_vals[name][0]-max_vals[name][2]) / max_vals[name][2]) * 100)
    if name in real_nets:
        edgecolor.append('red')
    else:
        edgecolor.append('none')

vmin = min( min(c_high), min(c_rand))
vmax = max( max(c_high), max(c_rand))


print('min: {} max: {}'.format(min(c_high), max(c_high)))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.set_title('HTR vs. High Degree')
ax1.set_xlabel(r'$\gamma$ (Estimated Degree Exponent)')
ax1.set_ylabel(r'$Q$ (Modularity)')
CMAP = get_cmap(cmocean.cm.thermal)
ax1.scatter(x, y, cmap=CMAP, c=c_high, vmin=vmin, vmax=vmax, edgecolor=edgecolor)
#fig.colorbar(label='% Performance Gain')
#plt.savefig('../results/plots/HTR_vs_high.png', dpi=100)


print('min: {} max: {}'.format(min(c_rand), max(c_rand)))

ax2.set_title('HTR vs. Random Degree')
ax2.set_xlabel(r'$\gamma$ (Estimated Degree Exponent)')
#ax2.set_ylabel(r'$Q$ (Modularity)')
res = ax2.scatter(x, y, cmap=CMAP, c=c_rand, vmin=vmin, vmax=vmax, edgecolor=edgecolor)

fig.colorbar(res, label='% Performance Gain', ax=[ax1, ax2])
#plt.tight_layout()
plt.savefig('../results/plots/HTR_limits_leiden.png', dpi=400)