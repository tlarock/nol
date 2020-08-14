#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:44:31 2020

@author: Stefan McCabe
@author: Timothy LaRock
"""
import igraph as ig
import leidenalg


def nx_to_ig(G):
    """ 
    Convert a NetworkX graph object to an igraph graph object.
    """
    H = ig.Graph()
    if isinstance(list(G.edges)[0][1], str):
        H.add_vertices([str(i) for i in G.nodes])
        H.add_edges([(str(i), str(j)) for i, j in G.edges])
    else:
        H.add_vertices([i for i in G.nodes])
        H.add_edges([(i, j) for i, j in G.edges])

    return H

    
    
def get_leiden_modularity(G):
    '''
    Accept a networkx graph, return the modularity of the best
    partition according to the Leiden algorithm.
    '''    
    Gi = nx_to_ig(G)
    leid = leidenalg.find_partition(Gi, leidenalg.ModularityVertexPartition)
    m1 = leid.modularity
    return m1