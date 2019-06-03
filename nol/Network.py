#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 08:57:23 2017

@author: sahely
"""
from collections import OrderedDict
from itertools import combinations_with_replacement
import networkx as nx
import numpy as np
import utility
import DefaultFeatures
import RefexFeatures
import Node2VecFeatures
import RefexAndNode2VecFeatures as RefAndNodeFeats
import KNNFeatures as knn
import NetworkDiscoveryFeatures as netdisc

class Network(object):
    """
    Network object contains the true network, the current version of the sample network,
    as well as the feature matrix for the sample network and associated functions.
    """

    def __init__(self, real_adj_list, sample_adj_list, \
                 calculate_features=True, feature_order='linear', feature_type='default', attribute_dict = None):
        self.feature_order = feature_order
        self.probed_node = []
        self.probedNodeSet = set()
        self.complete_graph_adjlist = real_adj_list     # full adjacency list
        self.complete_node_set = {k for k in real_adj_list for v in real_adj_list[k]}
        self.total_nodes = len(self.complete_node_set)
        self.sample_graph_adjlist = sample_adj_list    # sample adjacency list
        self.sample_adjlist_sets = self.get_sample_adjlist_sets()
        self.sample_node_set = set()
        for k in sample_adj_list:
            self.sample_node_set.add(k)
            for v in sample_adj_list[k]:
                self.sample_node_set.add(v)

        self.node_to_row, self.row_to_node = self.initialize_maps()
        self.initialize_sets()
        self.G = self.create_nx_graph()
        self.D = self.calculate_degree()
        self.feature_order = feature_order
        self.feature_type = feature_type
        self.original_node_set = self.sample_node_set.copy()
        self.rewardCounts = np.zeros((len(self.original_node_set),))

        ## attributes
        self.attribute_dict = attribute_dict
        if attribute_dict:
            for node in (self.complete_node_set - set(self.attribute_dict.keys())):
                self.attribute_dict[node] = list()

        ## Get the features functions
        if self.feature_type == 'default':
            self.calculate_features = DefaultFeatures.calculate_features
            self.update_features = DefaultFeatures.update_features
        elif self.feature_type == 'netdisc':
            self.calculate_features = netdisc.calculate_features
            self.update_features = netdisc.update_features
        elif self.feature_type == 'refex':
            self.calculate_features = RefexFeatures.calculate_features
            self.update_features = RefexFeatures.update_features
            self.egonets = dict() ## TODO initializing the dictionary of egonets
            self.egonet_edgecounts = dict() ## TODO initiliazing dictionary of egonet edgecounts
            RefexFeatures.set_egonets(self)
            self.F_no_normalization = None
        elif self.feature_type == 'node2vec':
            self.calculate_features = Node2VecFeatures.calculate_features
            self.update_features = Node2VecFeatures.update_features
        elif self.feature_type == 'n2v-refex':
            self.egonets = dict() ## TODO initializing the dictionary of egonets
            self.egonet_edgecounts = dict() ## TODO initiliazing dictionary of egonet edgecounts
            RefexFeatures.set_egonets(self)
            self.F_no_normalization = None
            self.calculate_features = RefAndNodeFeats.calculate_features
            self.update_features = RefAndNodeFeats.update_features
        elif self.feature_type == 'knn':
            self.egonets = dict() ## TODO initializing the dictionary of egonets
            self.egonet_edgecounts = dict() ## TODO initiliazing dictionary of egonet edgecounts
            knn.set_egonets(self)
            self.F_no_normalization = None
            self.calculate_features = knn.calculate_features
            self.update_features = knn.update_features

        if calculate_features:
            self.F = self.calculate_features(self, order=self.feature_order) # This line sets self.NumF

    def initialize_maps(self):
        """
        Initializes the node_to_row and row_to_node mapping dictionaries.
        """
        node_to_row = OrderedDict()
        row_to_node = OrderedDict()
        for row, node in enumerate(self.sample_node_set):
            node_to_row[node] = row
            row_to_node[row] = node
        return node_to_row, row_to_node

    def get_sample_adjlist_sets(self):
        """
        Converts the adjacency list dict of dicts into a dict of sets represnetation, which is
        faster to work with when we need quick set operations.
        """
        sample_adjlist_sets = dict()
        for node in self.sample_graph_adjlist.keys():
            sample_adjlist_sets[node] = set(self.sample_graph_adjlist[node].keys())
        return sample_adjlist_sets

    def initialize_sets(self):
        """
        Initializes probed neighbor set.
        """
        self.probed_neighbors = dict()
        #self.unprobed_neighbors = dict()
        for node in self.node_to_row.keys():
            self.probed_neighbors[node] = set() # keep track of probed neighbors (none yet)

    def calculate_degree(self):
        """
        Get degree of the nodes in the sample.
        NOTE: Relies on self.G, NOT self.sample_graph_adjlist or self.sample_adjlist_sets
        """
        sorted_nodes = self.node_to_row.keys()
        degrees = self.G.degree()
        self.D = np.array([degrees[u] for u in sorted_nodes])
        return np.nan_to_num(self.D)

    def get_degree(self):
        """
        Returns the degree nparray, self.D
        NOTE: Does not compute degree! Should already be computed before invoking this function.
        """
        return self.D

    def get_numfeature(self):
        return self.NumF


    def copy(self):
        #TODO Should update this to include self.G in the copy
        """
        Creates a new copy of the Network object.
        """
        sadjlist = dict()
        for node in self.sample_graph_adjlist:
            sadjlist[node] = self.sample_graph_adjlist[node].copy()

        return Network(self.complete_graph_adjlist, sadjlist, feature_type = self.feature_type)

    def create_nx_graph(self, graph='sample'):
        """
        Returns the networkx representation of the current Network object.
        NOTE: It is not necessary to recreate the object every time you need it. Access with self.G
        once it has been created.
        """
        if graph == 'sample':
            adjlist = self.sample_graph_adjlist
        else:
            adjlist = self.complete_graph_adjlist

        edgelist = set()
        for source, neighbors in adjlist.items():
            for target in neighbors:
                edgelist.add((source, target))

        G = nx.Graph()
        G.add_edges_from(edgelist)

        singletons = self.sample_node_set - set(G.nodes())
        G.add_nodes_from(singletons)

        return G

    def bin_feature(self,feature):
        indices = range(len(feature))
        indexed_feature = dict(zip(indices, feature))
        bins = utility.vertical_log_binning(indexed_feature)
        return np.array(list(bins.values()))

    def compute_avg_deg(self):
        """
        Compute and return average degree.
        """
        return sum(self.G.degree().values()) / float(self.G.number_of_nodes())

    def normalize_array(self,arr):
        """
        Accepts an np array, returns the normalized version.
        """
        # normalize values
        valid_indices = np.where(arr > 0)[0]
        if len(valid_indices) > 0:
            minimum = min(arr.min(), arr[valid_indices].min())
            maximum = arr[valid_indices].max()
            difference = maximum - minimum
            arr[valid_indices] = (arr[valid_indices]-minimum) / (difference)
        return arr

    def calculate_higher_order_features(self, new_feature_matrix, order='linear'):
        self.F = np.nan_to_num(new_feature_matrix)
        if order == 'cubic':
            self.F = np.insert(self.F, 0, values=1, axis=1)
            R = np.array(list(combinations_with_replacement(range(self.NumF+1), 3)))
            self.F = np.einsum('ij,ij,ij->ij', self.F[:, R[:, 2]], self.F[:, R[:, 1]], self.F[:, R[:, 0]])
            self.NumF = self.F.shape[1]
        elif order == 'quadratic':
            self.F = np.insert(self.F, 0, values=1, axis=1)
            R = np.triu_indices(self.NumF+1)
            self.F = np.einsum('ij,ij->ij', self.F[:, R[1]], self.F[:, R[0]])
            self.NumF = self.F.shape[1]

        return self.F


    def simple_probe(self, node):
        neighbors = self.complete_graph_adjlist[node]    # get node neighbors in real graph
        self.sample_graph_adjlist[node] = neighbors # add neighbors to probed node sample
        self.probed_node.append(node)
        self.probedNodeSet.add(node)

        neighbors = set(neighbors)
        self.sample_adjlist_sets[node] = neighbors
        # get the novel neighbors
        new_neighbors = [x for x in neighbors if x not in self.sample_node_set]
        self.sample_node_set.update(new_neighbors)
        edges = [(node, neighbor) for neighbor in neighbors]
        self.G.add_edges_from(edges)

        return new_neighbors

    def probe(self, node):
        """
        Probes a node. The operations include:
            - updating the adjacency lists (both the dict of
                dicts and dict of sets),
            - counting the number of new nodes discovered,
            - updating the probed/unprobed neighbors for each node,
            - updating the networkx graph and
            - updating the degree of affected nodes.
        Returns the novel nodes found.
        """
        ## get node neighbors in real graph
        neighbors = self.complete_graph_adjlist[node]

        ## list of all nodes that node is now connected to that they weren't before
        new_edges = set([u for u in neighbors if u not in self.sample_adjlist_sets[node]])

        ## add neighbors to sample graph and append probed node
        self.sample_graph_adjlist[node] = neighbors
        self.probed_node.append(node)
        self.probedNodeSet.add(node)
        neighbors = set(neighbors)
        self.sample_adjlist_sets[node] = neighbors

        ## get the novel neighbors (only nodes that were unobserved at t-1)
        ## and update the set of sample nodes, extend degree vector
        new_neighbors = [x for x in neighbors if x not in self.sample_node_set]
        self.sample_node_set.update(new_neighbors)
        self.D = np.concatenate((self.D, np.zeros(len(new_neighbors))))
        self.D[self.node_to_row[node]] = len(neighbors)
        ## Extend rewardCounts. This value won't change for the probed node.
        self.rewardCounts = np.concatenate((self.rewardCounts, np.zeros(len(new_neighbors))))

        edges = []
        i = max(self.row_to_node.keys()) + 1
        for neighbor in neighbors:
            self.probed_neighbors.setdefault(neighbor, set())
            if neighbor not in self.node_to_row.keys():
                self.node_to_row[neighbor] = i
                self.row_to_node[i] = neighbor
                i += 1

            ## update the reward count for this neighbour
            ## The idea behind this is to remove order effects
            ## in rewards: If Tim is probed before Tina and they have
            ## a common neighbor, Tim will get the reward and Tina won't, 
            ## even though she _could_ have if she had been probed first!
            if neighbor in new_edges:
                self.rewardCounts[self.node_to_row[neighbor]] += 1

            self.sample_graph_adjlist.setdefault(neighbor, dict())
            self.sample_adjlist_sets.setdefault(neighbor, set())
            self.sample_graph_adjlist[neighbor][node] = None # add node to neighbor's adjlist
            self.sample_adjlist_sets[neighbor].add(node)
            #self.sample_node_set.add(neighbour) # add neighbor to sample nodes if necessary
            #self.unprobed_neighbors[node].add(neighbour)
            # update NX graph
            edges.append((node, neighbor))
            # update degree
            deg = len(self.sample_graph_adjlist[neighbor])
            self.D[self.node_to_row[neighbor]] = deg

            # update probed neighbors
            self.probed_neighbors.setdefault(neighbor, set())
            self.probed_neighbors[neighbor].add(node)

        if 'refex' in self.feature_type or self.feature_type == 'knn':
            ## TODO update egonet
            tmp_nodes = self.sample_adjlist_sets[node].copy()
            tmp_nodes.add(node)
            nodes_to_update = set(tmp_nodes)
            for u in tmp_nodes:
                nodes_to_update.update(self.sample_adjlist_sets[u])

            RefexFeatures.set_egonets(self, nodes=list(nodes_to_update))

        # update NX graph
        self.G.add_edges_from(edges)
        return new_neighbors

    def get_index_of_probed_nodes(self):
        """
        Returns the indices in the feature matrix for the probed nodes.
        """
        return [i for i in self.row_to_node.keys() if self.row_to_node[i] in self.probed_node]

    def get_adjlist(self, node):
        """
        Return list of neighbors of node
        """
        if node in self.sample_graph_adjlist:
            adjn = list(self.sample_adjlist_sets[node])
        else:
            adjn = []

        return adjn

def main():
    np.set_printoptions(precision=2)
    test_complete_file = "../data/toys/toy1.txt"
    test_sample_file = "../data/toys/toy_comps.txt"
    #test_sample_file = "../data/toys/toy_comps_with_unobserved.txt"
    gsamp = nx.to_dict_of_dicts(nx.read_adjlist(test_sample_file))
    gcomp = nx.to_dict_of_dicts(nx.read_adjlist(test_complete_file))
    #net = Network(gcomp, gsamp, feature_type='default')
    net = Network(gcomp, gsamp, feature_type='knn')
    #net = Network(gcomp, gsamp, feature_type='node2vec')
    #net = Network(gcomp, gsamp, feature_type='n2v-refex')

    order = 'linear'
    feat = net.calculate_features(net, order=order)
    print(feat)
    print(net.row_to_node)
    net.probe('9')
    print("Probed node 9...")
    feat = net.update_features(net, '9', order=order)
    print(feat)
    print(net.row_to_node)
    net.probe('12')
    print("Probed node 12...")
    feat = net.update_features(net, '12', order=order)
    print(feat)
    print(net.row_to_node)
    net.probe('11')
    print("Probed node 11...")
    feat = net.update_features(net, '11', order=order)
    print(feat)
    print(net.row_to_node)
    print(net.calculate_features(net, order=order))

if __name__ == "__main__":
    main()
