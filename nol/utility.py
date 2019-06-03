#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:51:25 2017

@author: sahely
"""
import numpy as np
import random
import math
import operator

def read_attributes(filename, delim='\t', ids = []):
    """
    Read attributes from filename.
    Attribute file should be 1 row per attribute and
    columns corresponding to nodes with that attribute.
    """
    attribute_dict = dict()
    with open(filename, 'r') as f:
        for attr_id, line in enumerate(f):
            ## ignore any attributes not in 'ids' list (if given)
            if len(ids) > 0 and attr_id not in ids:
                continue

            L = line.split(delim)
            for node in L:
                attribute_dict.setdefault(node, [])
                attribute_dict[node].append(attr_id)

    return attribute_dict

def softmax(x):
    """
    softmax of a vector x
    """
    e_x = np.exp(x)
    return e_x / sum(e_x)

def getProbRestart():
    return 0.15

def random_pick(values,num):
    item_list= -1*np.ones([num]).astype(int)
    if len(values[values>0]) > 0:
        values[values<0] = 0
    else:
        values = np.ones(len(values))
    #values = np.exp(values)
    sumvals = sum(values)
    minval = min(values)
    if minval<0:
        pmf=(values-minval)/sumvals
    else:
        pmf=values/sumvals
    numvals = len(pmf)
    indices = np.arange(numvals)
    pmf=np.reshape(pmf, (pmf.shape[0],))
    for i in range(num):
        item_list[i] = np.random.choice(indices, p=pmf)

    return item_list


def read_edge_list(filename, delim = " "):
    nodes = set()
    from collections import defaultdict
    adjlist = defaultdict(set)
    with open(filename, 'r') as f:
        for line in f:
            edges = line.split(delim)
            source = edges[0].strip()
            target = edges[1].strip()
            adjlist[source].add(target)
            adjlist[target].add(source)
            nodes.add(source)
            nodes.add(target)

    num_nodes = len(nodes)

    return num_nodes, adjlist

# Xindi's vertical log binning code
def vertical_log_binning(data, p=0.5):
    index, value = zip(*sorted(data.items(), key=operator.itemgetter(1)))
    bin_result = []
    value = list(value)
    i = 1
    while len(value) > 0:
        num_to_bin = int(math.ceil(p*len(value)))
        edge_value = value[num_to_bin-1]
        to_bin = list(filter(lambda x: x <= edge_value, value))
        bin_result += [i]*len(to_bin)
        value = list(filter(lambda x: x > edge_value, value))
        i += 1
    bin_result_dict = dict(zip(index, bin_result))

    return bin_result_dict

#TODO: This function works for both edgelist and adjacency list. It's about the same speed as the nx
# function, doesn't require converting to dict of dicts. 
def read_network(filename):
    adjlist = dict()
    edges = set()
    nodes = set()

    ## open the file
    with open(filename, 'r') as f:

        ## read every line
        for line in f:
            ## ignore comments
            if '#' in line:
                continue

            ## strip the line and split on spaces
            s = line.strip().split()

            ## add every node to the set of nodes
            nodes.add(s[0])
            ## set a dictionary if there isn't one already
            adjlist.setdefault(s[0],dict())

            ## for every subsequent neighbor
            for i in range(1, len(s)):
                ## Once we see a {, we know the line is done (for nx edgelists)
                if '{' in s[i]:
                    break

                ## set the current neighbor
                adjlist[s[0]][s[i]] = dict()

                ## add the neighbor to the dictionary
                adjlist.setdefault(s[i],dict())
                adjlist[s[i]][s[0]] = dict()

                ## add node to nodes and edge to edges
                nodes.add(s[i])
                if (s[i],s[0]) not in edges:
                    edges.add((s[0],s[i]))

    return adjlist, nodes, edges
