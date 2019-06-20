#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:04:49 2017

@author: Tim S
"""

import os, sys, numpy as np
import networkx as nx


def generate_sample(G, sampling_method, p=0.01, attribute_dict=None, target_att=None, k=5, interval=500, max_tries=5):
    """
    Args:
        G: Adjacency list representation of the graph
        sampling_method: sampling method to use
        p: what portion of the network to sample, between 0 (nothing) and 1 (whole network)
    """
    if sampling_method.lower() not in ['node','iKNN', 'netdisc', 'walk']:
        raise ValueError("generateNodeSample: sampling_method is not one of 'node', 'edge', 'walk', or 'walkjump'.  No sample was generated.")
    if p < 0 or p > 1:
        raise ValueError("generateNodeSample: portion of network to sample invalid (must be between 0 and 1).  No sample was generated.")

    if sampling_method.lower() == 'node':
        return node_sample(G, p, interval, max_tries)
    elif sampling_method.lower() == 'netdisc':
        return netdisc_sample(G, attribute_dict, target_att, k)
    elif sampling_method.lower() == 'iknn':
        return node_sample_iKNN(G, p)
    elif sampling_method.lower() == 'walk':
        return generateWalkJumpSample(G, p, 0.15)


def node_sample_iKNN(adjacency_list, p):
    """ Generate what the Murai et al. paper refers to as a random node sample (a random walk with
    jum probability 1), or a random neighbor sample"""
    sample_adjlist = dict()
    sample_nodes = set()
    sample_edges = set()
    size = len(adjacency_list) * p

    v = np.random.choice(list(adjacency_list.keys()))
    sample_nodes.add(v)
    sample_adjlist.setdefault(v, dict())

    while len(sample_nodes) < size:
        if np.random.random() < 0.99:
            w = np.random.choice(list(adjacency_list[v]))
        else:
            w = np.random.choice(list(adjacency_list.keys()))

        sample_nodes.add(w)
        sample_adjlist.setdefault(w, dict())
        sample_adjlist[v][w] = dict()
        sample_adjlist[w][v] = dict()
        sample_edges.add((v,w))
        sample_edges.add((w,v))

        ## jump if v a singleton
        while len(list(adjacency_list[w])) < 1:
            w = np.random.choice(list(adjacency_list[w]))

        v = w

    return sample_adjlist, sample_nodes, sample_edges

def netdisc_sample(adjacency_list, attribute_dict, target_att, k):
    nodes = set()
    edges = set()
    sample_adjlist = dict()

    ## get the labeled nodes
    anomalous_nodes = [node for node, attr_lst in attribute_dict.items() if target_att in attr_lst]
    ## sample from the anomalous nodes
    sampled_anomalous = np.random.choice(anomalous_nodes, k, replace=False)
    ## add anomalous nodes and their neighbors to sample
    for node in sampled_anomalous:
        sample_adjlist.setdefault(node, dict())
        nodes.add(node)
        for ne in adjacency_list[node].keys():
            sample_adjlist.setdefault(ne, dict())
            sample_adjlist[node][ne] = dict()
            sample_adjlist[ne][node] = dict()
            nodes.add(ne)
            edges.add((ne,node))
            edges.add((node,ne))

    return sample_adjlist, nodes, edges

def node_sample(adjacency_list, p, interval, max_tries):
    def induce_subgraph(adjacency_list, nodes):
        sample_adjacency = dict()
        edges = set()
        for node in nodes:
            sample_adjacency.setdefault(node, dict())
            for ne in nodes.intersection(adjacency_list[node]):
                if (node, ne) in edges or (ne, node) in edges:
                    continue
                sample_adjacency[node][ne] = dict()
                sample_adjacency.setdefault(ne, dict())
                sample_adjacency[ne][node] = dict()
                edges.add((node, ne))

        return sample_adjacency, edges

    number_of_edges = sum([len(adjacency_list[u]) for u in adjacency_list.keys()]) / 2
    desired_num_edges = int(p * number_of_edges)

    i=1
    tries = 0
    sample_num_edges = 0
    while sample_num_edges < desired_num_edges:
        sampled_nodes = set(np.random.choice(list(adjacency_list.keys()),interval*i, replace=False))
        sample_adjlist, edges = induce_subgraph(adjacency_list, sampled_nodes)
        sample_num_edges = len(edges)
        i=i+1

        ## make sure the sample isn't too big! (more than 20%, arbitrarily)
        if sample_num_edges > (desired_num_edges + desired_num_edges * 0.2):
            sample_num_edges = 0
            if np.random.random() > 0.5:
                i = i - 1
            else:
                i = i - 2
            tries += 1

        if tries == max_tries:
            interval = int(interval/2.0)
            tries = 0

    return sample_adjlist, sampled_nodes, edges

def generateWalkJumpSample(adjacency_list, p, jump_probability):
    """
    Generate random walk sample from real graph
    """
    sample_network = dict()
    nodes = set()
    edges = set()
    number_of_edges = sum([len(adjacency_list[u]) for u in adjacency_list.keys()]) / 2
    desired_num_edges = int(p * number_of_edges)

    currNode = np.random.choice(list(adjacency_list.keys()))
    nodes.add(currNode)
    sample_network[currNode] = dict()
    while len(edges) < desired_num_edges:
        if (len(adjacency_list[currNode]) > 0) and (np.random.random() > jump_probability):
            # if we have somewhere to walk to
            nextNode = np.random.choice(list(adjacency_list[currNode].keys()))
            sample_network.setdefault(nextNode, dict())
            sample_network[nextNode][currNode] = dict()
            sample_network[currNode][nextNode] = dict()
            if (currNode, nextNode) not in edges:
                edges.add((currNode, nextNode))
            nodes.add(nextNode)
            currNode = nextNode
        else:
            ## jump
            nextNode = np.random.choice(list(adjacency_list.keys()))
            sample_network[nextNode] = dict()
            nodes.add(nextNode)
            currNode = nextNode

    return sample_network, nodes, edges

