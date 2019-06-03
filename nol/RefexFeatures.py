import numpy as np
from DefaultFeatures import compute_connected_components

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
    ## TODO Compute connected components
    connected_component_sizes = compute_connected_components(self, compute_nx=True)

    num_recursive_feats = len(recursive_features[0]) + 1

    features = np.zeros( (len(self.node_to_row), num_recursive_feats) )
    for node, row in self.node_to_row.items():
        features[row] = np.concatenate((recursive_features[row], np.array([connected_component_sizes[row]])))

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
    # TODO Update components
    extension_length = recursive_features.shape[0] - self.F_no_normalization[:,num_recursive_feats-1].shape[0]
    component_size_array = np.concatenate((self.F_no_normalization[:,num_recursive_feats-1], np.full(extension_length, -1, dtype=int)))

    connected_component_sizes = update_components(self, component_size_array, node)

    ## get the number of new nodes by comparing the old feature table
    old_length = self.F_no_normalization.shape[0]
    extension_length = len(self.node_to_row.keys()) - old_length
    ## Concatentate 0s to the feature matrix
    self.F_no_normalization = np.concatenate( (self.F_no_normalization, np.zeros( (extension_length, num_recursive_feats) )))

    ## update the feature tables
    for node in nodes_to_update:
        row = self.node_to_row[node]
        self.F_no_normalization[row] = np.concatenate((recursive_features[row], np.array([connected_component_sizes[row]])))

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
            neighborhood_features = neighborhood_features[:,0:4]
        else:
            neighborhood_features = self.F_no_normalization[:,0:4]
    else:
        neighborhood_features = np.zeros((len(self.node_to_row), 4))


    for node in nodes:
        egonet = self.egonets[node]
        row = self.node_to_row[node]

        ## egonet degree (same as regular degree)
        degree = len(egonet[node])
        self.D[row] = degree
        ## triangles (number of edges between nodes in the ego net) 
        ## NOTE: Double counting in self.set_egonets so dividing by 2 (assuming undirected)
        triangles = 0
        if self.egonet_edgecounts[node]['within'] > degree:
            triangles = (self.egonet_edgecounts[node]['within'] - degree) / 2

        ## Edges leaving the egonet: total degree - egonet degree for every neighbor
        edges_leaving = self.egonet_edgecounts[node]['leaving']

        number_probed_neighbors = len(self.probed_neighbors[node])
        ## set features
        #neighborhood_features[row] = np.array([degree, triangles, edges_leaving, number_probed_neighbors, self.rewardCounts[row]])
        neighborhood_features[row] = np.array([degree, triangles, edges_leaving, number_probed_neighbors])

    return neighborhood_features

def compute_recursive_features(self, curr_features, nodes=None):
    """
    Compute recursive features (averages and sums).
    Returns dictionary of current features appended with recursive features.
    """
    if nodes is None:
        nodes = self.sample_graph_adjlist.keys()
    #new_features = dict(curr_features)
    new_features = np.hstack((curr_features, np.zeros((curr_features.shape[0], curr_features.shape[1]*4))))
    ## for each node
    for node in nodes:
        ## get the egonet node feature matrix
        egonet = self.egonets[node]
        row = self.node_to_row[node]
        neighbor_indices = [self.node_to_row[u] for u in egonet.keys() if u != node]
        if len(neighbor_indices) > 0:
            ## compute means/sums
            sums = np.sum(curr_features[neighbor_indices,:], axis = 0)
            averages = np.median(curr_features[neighbor_indices,:], axis = 0)
            mins = np.min(curr_features[neighbor_indices,:], axis = 0)
            maxs = np.max(curr_features[neighbor_indices,:], axis = 0)
        else:
            ## if there are no neighbors, all features are 0
            sums = np.zeros(curr_features.shape[1])
            averages = sums
            mins = sums
            maxs = sums

        ## add as features
        recursive_features = np.concatenate( (sums, averages, mins, maxs))
        new_features[row] = np.concatenate( (curr_features[row], recursive_features) )

    return new_features


def update_components(self, connected_component_sizes, probed_node):
    """
    Update the features of the neighbors of the probed node.
    For each neighbor, updates clustering, then checks components,
    then fraction of probed neighbors.
    """
    probed_node_row = self.node_to_row[probed_node]
    components_to_merge = set()
    probed_c = self.components[probed_node_row]
    for neighbor in self.sample_graph_adjlist[probed_node]:
        neighbor_row = self.node_to_row[neighbor]

        # if the new neighbor was previously in a different component, must recompute
        if neighbor_row < len(self.components) and \
            self.components[neighbor_row] != probed_c:
            components_to_merge.add(self.components[neighbor_row])
        else:
            # if new neighbor is already in the same component/ in no component, just add to the
            # probed node's component
            if neighbor_row >= len(self.components):
                update_len = ((neighbor_row+1) - len(self.components))
                self.components = np.concatenate((self.components, np.full((update_len, ), probed_c, dtype=int)))
                self.node_component_sizes = np.concatenate((self.node_component_sizes, np.zeros((update_len), dtype=int)))
            self.components[neighbor_row] = probed_c
            self.connected_components[probed_c].add(neighbor)
            self.connected_component_sizes[probed_c] = len(self.connected_components[probed_c])
            self.node_component_sizes[np.where(self.components == probed_c)] = self.connected_component_sizes[probed_c]

    # UPDATE FEATURE 3
    # Need to recompute this every time no matter what, but if recompute_components is true then
    # the components will actually be recomputed (i.e. nx.connected_components)
    new_component_sizes = compute_connected_components(self, False, probed_node, components_to_merge, connected_component_sizes)

    return new_component_sizes
