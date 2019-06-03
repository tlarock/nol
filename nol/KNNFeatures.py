import numpy as np
def set_egonets(self, nodes = None):
    """
    Updates the self.egonets data structure, which is a
    dictionary indexed by node pointing to the induced subgraph
    on the node and its neighbors. Also updates the egonet_edgecounts
    for each node, used in a calculation later.
    """
    if nodes is None:
        nodes = self.sample_graph_adjlist.keys()

    ## set the egonet for each node
    for node in nodes:
        #if node not in self.egonets.keys():
        egonet = dict()
        ## start with the neighbors of the node
        egonet[node] = self.sample_graph_adjlist[node]
        leaving_edgecount = 0
        egonet_nodes = self.sample_adjlist_sets[node]
        within_edgecount = len(egonet_nodes)
        #else:
        #    egonet = self.egonets[node]
        #    leaving_edgecount = self.egonet_edgecounts[node]['leaving']
        #    within_edgecount = self.egonet_edgecounts[node]['within']
        #    egonet_nodes = egonet.keys() - self.sample_adjlist_sets[node]
        #assert node not in egonet_nodes, 'node is in egonet_nodes!'
        #print('node: ' + str(node) + ' neighbors: ' + str(egonet_nodes))
        ## for every neighbor, add links to nodes that are in the egonet
        for neighbor in egonet_nodes:
            if neighbor in self.sample_graph_adjlist.keys():
                neighbors_of_neighbor = self.sample_adjlist_sets[neighbor]
                egonet_neighbors = neighbors_of_neighbor.intersection(egonet_nodes)
                egonet[neighbor] = {key:dict() for key in egonet_neighbors}

                ## update # of edges within the egonet
                within_edgecount += len(egonet_neighbors)

                ## update # of edges leaving the egonet
                if (len(self.sample_adjlist_sets[neighbor]) - len(egonet_neighbors)) > 0:
                    leaving_edgecount += len(self.sample_adjlist_sets[neighbor]) - len(egonet_neighbors) - 1
                #print('neighbor: ' + str(neighbor) + ' current within edgecount: ' + str(within_edgecount) + ' current leaving edgecount: ' + str(leaving_edgecount) + ' egonet: ' + str(egonet[neighbor]))
            else:
                egonet[neighbor] = {node:dict()}
        ## set the egonet
        self.egonets[node] = dict(egonet)
        ## set edgecount properties
        self.egonet_edgecounts[node] = {'within':within_edgecount, 'leaving':leaving_edgecount}
        #print('edgecounts: ' + str(self.egonet_edgecounts[node]))

def calculate_features(self, order='linear'):
    """
    Calculates features from scratch. Use update_features for updates after a probe!
    Using recursive features following ReFeX:
        nodal: degree
        egonet: triangles (# edges within), # edges going out, fraction probed neighbors
        recursive: sums and averages of these features
    """
    ## compute neighborhood features
    neighborhood_features = compute_neighborhood_features(self)
    ## compute recursive features
    recursive_features = compute_recursive_features(self, neighborhood_features)

    num_recursive_feats = len(recursive_features[0])

    features = np.zeros( (len(self.node_to_row), num_recursive_feats) )
    for node, row in self.node_to_row.items():
        features[row] = recursive_features[row]

    ## store the non normalized features
    self.F_no_normalization = features.copy()
    ## normalize by the max
    max_feats = np.max(self.F_no_normalization, axis = 0)
    min_feats = np.min(self.F_no_normalization, axis = 0)
    normalization = max_feats - min_feats
    normalization[normalization == 0] = 1
    features = (self.F_no_normalization - min_feats) / normalization

    self.NumF = features.shape[1]
    return features

def update_features(self, node, order='linear'):
    """
    Updates the feature matrix based on the node being probed.
    """
    ## get the nodes to update
    tmp_nodes = self.sample_adjlist_sets[node].copy()
    tmp_nodes.add(node)
    nodes_to_update = set(tmp_nodes)
    for u in tmp_nodes:
        nodes_to_update.update(self.sample_adjlist_sets[u])
    #nodes_to_update = list(self.sample_graph_adjlist.keys()) + [node]
    ## compute the features for these nodes
    neighborhood_features = compute_neighborhood_features(self, nodes_to_update)
    recursive_features = compute_recursive_features(self, neighborhood_features, nodes_to_update)
    num_recursive_feats = self.F_no_normalization.shape[1]

    ## get the number of new nodes by comparing the old feature table
    old_length = self.F_no_normalization.shape[0]
    extension_length = len(self.node_to_row.keys()) - old_length
    ## Concatentate 0s to the feature matrix
    self.F_no_normalization = np.concatenate( (self.F_no_normalization, np.zeros( (extension_length, num_recursive_feats) )))

    ## update the feature tables
    for node in nodes_to_update:
        row = self.node_to_row[node]
        self.F_no_normalization[row] = recursive_features[row]

    ## normalize by the max
    max_feats = np.max(self.F_no_normalization, axis = 0)
    min_feats = np.min(self.F_no_normalization, axis = 0)
    normalization = max_feats - min_feats
    normalization[normalization == 0] = 1
    features = (self.F_no_normalization - min_feats) / normalization

    assert features.shape == self.F_no_normalization.shape, 'features.shape != no norm.shape!'
    self.F = features
    self.NumF = features.shape[1]
    return features

def compute_neighborhood_features(self, nodes = None):
    """
    Returns a dictionary indexed by node id of local features, both
    nodal and egonet.
    """
    if nodes is None:
        nodes = self.sample_graph_adjlist.keys()

    ## neighborhood_features will be a dict of lists
    ## if there are features already
    if self.F_no_normalization is not None:
        ## initialize the features from the unnormalized features
        if len(self.node_to_row) != self.F_no_normalization.shape[0]:
            extension_length = len(self.node_to_row) - self.F_no_normalization.shape[0]
            neighborhood_features = np.vstack( (self.F_no_normalization, np.zeros( (extension_length, self.F_no_normalization.shape[1]))))
            neighborhood_features = neighborhood_features[:,0:2]
        else:
            neighborhood_features = self.F_no_normalization[:,0:2]
    else:
        neighborhood_features = np.zeros((len(self.node_to_row), 2))


    for node in nodes:
        egonet = self.egonets[node]
        row = self.node_to_row[node]

        ## egonet degree (same as regular degree)
        degree = len(egonet[node])
        self.D[row] = degree
        number_probed_neighbors = len(self.probed_neighbors[node])
        ## set features
        neighborhood_features[row] = np.array([degree, number_probed_neighbors])

    return neighborhood_features

def compute_recursive_features(self, curr_features, nodes=None):
    """
    Compute recursive features (averages and sums).
    Returns dictionary of current features appended with recursive features.
    """
    if nodes is None:
        nodes = self.sample_graph_adjlist.keys()
    new_features = np.hstack((curr_features, np.zeros((curr_features.shape[0], curr_features.shape[1]))))
    ## for each node
    for node in nodes:
        ## get the egonet node feature matrix
        egonet = self.egonets[node]
        row = self.node_to_row[node]
        neighbor_indices = [self.node_to_row[u] for u in egonet.keys() if u != node]
        if len(neighbor_indices) > 0:
            ## compute means/sums
            median = np.array(np.median(curr_features[neighbor_indices,0], axis = 0))
            mean = np.array(np.mean(curr_features[neighbor_indices,0], axis = 0))
        else:
            ## if there are no neighbors, all features are 0
            mean = 0
            median = 0

        ## add as features
        recursive_features = np.array([mean, median])
        new_features[row] = np.concatenate( (curr_features[row], recursive_features) )

    return new_features
