import networkx as nx
import numpy as np


def calculate_features(self, order='linear'):
    """
    Calculates features from scratch. Use update_features for updates after a probe!
    """
    sorted_nodes = self.node_to_row.keys()
    # FEATURE 1 in sample degree
    deg = self.D.copy()
    deg = (deg-deg.min())/(deg.max()-deg.min())

    # FEATURE 2 in sample clustering coefficient
    # get clustering coefficient
    cc = nx.clustering(self.G)
    cc_array = np.array([cc[u] for u in sorted_nodes])

    # FEATURE 3 Normalized size of my connected component
    connected_component_sizes = compute_connected_components(self, compute_nx=True)

    # FEATURE 4 fraction of neighbors probed
    frac_probed_neighbors = fraction_probed_neighbors(self)

    if self.attribute_dict is None:
        new_feature_matrix = np.column_stack((deg, cc_array, connected_component_sizes, frac_probed_neighbors))
    else:
        num_target_neighbors = number_target_neighbors(self)
        #frac_target_neighbors = num_target_neighbors / self.D
        new_feature_matrix = np.column_stack((deg, cc_array, connected_component_sizes, frac_probed_neighbors, num_target_neighbors))

    ## concatenate the new matrix
    self.NumF = new_feature_matrix.shape[1]

    if order != 'linear':
        self.F = self.calculate_higher_order_features(new_feature_matrix, order=order)
    else:
        self.F = new_feature_matrix

    return self.F

def update_features(self, probed_node, order='linear'):
        """
        Updates the feature matrix (self.F) for the nodes in the current sample.
        If no probed_node is specified, just calculates features from scratch.
        If probed_node is specified, just updates nodes whose features could change based on the
        probe.
        """
        # From  May 23 meeting, features are:
        #   1. nodes degree in sample
        #   2. nodes clustering coefficient in sample
        #   3. Normalized size of the connected component that I'm in
        #   4. Fraction of node's neighbors which have been probed
        #   5. intersection of node's 1-hop neighbors with probed node 1-hop neighbors
        #   6. Fraction of node's 2-hop neighbors which have been probed (normalized by # 2-hop neighbors)
        #   7. Fraction of probed nodes reachable in 2 hops (normalized by # probed nodes)
        # if a node has been probed, add a new row of features for that
        # node, then update ONLY the neighbors of that node

        probed_node_row = self.node_to_row[probed_node]
        # FEATURE 1 in sample degree
        deg = self.D.copy()
        deg = (deg-deg.min())/(deg.max()-deg.min())
        # Compute new shape
        old_length = len(self.F[:, 1])
        extension_length = len(self.node_to_row.keys()) - old_length

        cc_array = np.concatenate((self.F[:, 1], np.zeros(extension_length)))
        # get clustering of probed node, set adjacenct to probed feature
        clustering_nodes = self.sample_adjlist_sets[probed_node].copy()
        clustering_nodes.add(probed_node)
        clustering = nx.clustering(self.G, clustering_nodes)
        cc_array[probed_node_row] = clustering[probed_node]
        component_size_array = np.concatenate((self.F[:, 2], np.full(extension_length, -1, dtype=int)))
        frac_probed_array = np.concatenate((self.F[:, 3], np.zeros(extension_length)))
        number_target_neighbors = np.concatenate((self.F[:,4], np.zeros(extension_length)))

        # update fraction of probed neighbors of probed node
        if probed_node in self.probed_neighbors.keys():
            if len(self.sample_graph_adjlist[probed_node]) > 0:
                frac_probed_array[probed_node_row] = len(self.probed_neighbors[probed_node]) / \
                    float(len(self.sample_graph_adjlist[probed_node]))
        else:
            frac_probed_array[probed_node_row] = 0.0

        ## FEATURE 6 (attributes) TODO Assumes 0 is label!!
        if self.attribute_dict is not None:
            number_target_neighbors[probed_node_row] = sum([1 for node in self.sample_adjlist_sets[probed_node] if 0 in self.attribute_dict[node]])
            new_feature_matrix = np.column_stack((deg, cc_array, component_size_array, frac_probed_array, number_target_neighbors))
        else:
            new_feature_matrix = np.column_stack((deg, cc_array, component_size_array, frac_probed_array))

        # Update neighbor features
        update_neighbors(self, new_feature_matrix, probed_node, clustering)

        self.NumF = new_feature_matrix.shape[1]
        if order != 'linear':
            self.F = self.calculate_higher_order_features(new_feature_matrix, order=order)
        else:
            self.F = new_feature_matrix

        return self.F


def update_neighbors(self, new_feature_matrix, probed_node, clustering):
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

        # UPDATE FEATURE 2 (clustering)
        new_feature_matrix[neighbor_row, 1] = clustering[neighbor] #self.get_cc(neighbor)
        # UPDATE FEATURE 3 TEST
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

        # UPDATE FEATURE 4 (frac probed neighbors)
        denom = float(len(self.sample_graph_adjlist[neighbor]))
        if denom > 0:
            fraction_neighbors_probed = float(len(self.probed_neighbors[neighbor])) / denom
        else:
            fraction_neighbors_probed = 0.0
        new_feature_matrix[neighbor_row, 3] = fraction_neighbors_probed

        ## UPDATE FEATURE 6 (number target neighbors) TODO Assumes 0 label
        if self.attribute_dict is not None:
            if 0 in self.attribute_dict[probed_node]:
                new_feature_matrix[neighbor_row, 4] += 1

    # UPDATE FEATURE 3
    # Need to recompute this every time no matter what, but if recompute_components is true then
    # the components will actually be recomputed (i.e. nx.connected_components)
    connected_component_sizes = compute_connected_components(self, False, probed_node, components_to_merge, new_feature_matrix[:, 2])
    new_feature_matrix[:, 2] = connected_component_sizes


def fraction_probed_neighbors(self):
    """
    Computes the fraction of all probed nodes connected to each node.

    NOTE: As of 1/22, we decided to change the normalization to # probed neighbors / # probed nodes,
    rather than # probed neighbors / # neighbors. Check previous git commits.
    """

    sorted_nodes = self.node_to_row.keys()
    if len(self.probedNodeSet) == 0:
        return np.array([0 for node in sorted_nodes])

    fraction_probed_neighbors = np.array([len(self.probed_neighbors[node]) / len(self.sample_adjlist_sets[node]) for node in sorted_nodes])
    #fraction_probed_neighbors = probed_neighbors / len(self.probedNodeSet)
    return fraction_probed_neighbors



def number_target_neighbors(self):
    """
    Computes the number of nodes with target attribute attached to each node

    """

    sorted_nodes = self.node_to_row.keys()
    ## TODO This is ad-hoc and assumes 0 is the target label!!
    target_nodes = set([node for node in sorted_nodes if 0 in self.attribute_dict[node]])
    number_target_neighbors = np.array([len(target_nodes.intersection(self.sample_adjlist_sets[node])) for node in sorted_nodes])
    #fraction_probed_neighbors = probed_neighbors / len(self.probedNodeSet)
    return number_target_neighbors

def intersection_probed_neighbors(self, neighbors_of_probed, nodes_to_update=None, current_vals=None):
    """
    Computes the intersection between a node's neighbors and the neighbors of probed nodes.
    """
    num_neighbors_of_probed = float(len(neighbors_of_probed))
    if num_neighbors_of_probed > 0:
        if nodes_to_update is None:
            nodes_to_update = self.node_to_row.keys()
        if current_vals is None:
            ret = np.zeros(len(nodes_to_update))
        else:
            ret = current_vals
        for node in nodes_to_update:
            neighbors = self.sample_adjlist_sets[node]
            probed_neighbors = float(len(neighbors.intersection(neighbors_of_probed)))
            ret[self.node_to_row[node]] = probed_neighbors / num_neighbors_of_probed
    else:
        ret = np.zeros(len(self.node_to_row.keys()))

    return ret

def compute_connected_components(self, compute_nx=True, probed_node=None, comps_to_merge=None, current_norm_vals=None):
    """
    Computes the NORMALIZED connected components of the selfwork.
    If compute_nx is True, actually computes components from scratch using selfworkx.
    Otherwise, we update self.connected_components, self.connected_component_sizes, and
    self.components based on comps_to_merge.
    UPDATE: As of 7/25, we now keep track of node_component_sizes, i.e. the size of the
    component each node is in. This is managed in BOTH update_neighbors AND
    compute_connected_components. It is an np array with entries corresponding to rows in the
    feature matrix.
    Returns np array of normalized component sizes.
    """
    if compute_nx:
        # compute the nx components
        self.connected_components = {k:c for k, c in enumerate(nx.connected_components(self.G))}
        connected_components = self.connected_components
        self.components = np.zeros((len(self.node_to_row)), dtype=int)
        self.connected_component_sizes = dict()
        self.node_component_sizes = np.zeros((len(self.node_to_row)))
        # initialize min/max
        min_component_size = float('inf')
        max_component_size = 0
        # loop sets self.connected_component_sizes, self.components, max/min
        for i, component in self.connected_components.items():
            size = len(component)
            self.connected_component_sizes[i] = size
            if size < min_component_size:
                min_component_size = size
            if size > max_component_size:
                max_component_size = size

            for node in component:
                self.components[self.node_to_row[node]] = i
                self.node_component_sizes[self.node_to_row[node]] = size
    else:
        # No need to compute in this case
        connected_components = self.connected_components
        # For each component:
        min_component_size = self.min_comp_size
        max_component_size = self.max_comp_size
        probed_comp = self.components[self.node_to_row[probed_node]]
        if comps_to_merge is None:
            comps_to_merge = []

        for comp in comps_to_merge:
            # Keep probed_node's component (arbitrary choice),
            # add all of each other component's nodes + size to probed component
            self.connected_components[probed_comp].update(self.connected_components[comp])
            self.connected_component_sizes[probed_comp] = len(self.connected_components[probed_comp])
            for node in self.connected_components[comp]:
                self.components[self.node_to_row[node]] = probed_comp
            # pop the old component from the dictionaries
            self.connected_components.pop(comp)
            self.connected_component_sizes.pop(comp)

            # update probed component size across the board
            self.node_component_sizes[np.where(self.components == probed_comp)] = self.connected_component_sizes[probed_comp]
            # If the min/max size changed, will need to recompute normalized value for ALL
            # components, rather than just the probed node's component.
            new_min = min(self.connected_component_sizes.values())
            if self.connected_component_sizes[probed_comp] > max_component_size:
                new_max = self.connected_component_sizes[probed_comp]
            else:
                new_max = max_component_size
            if new_min != min_component_size or new_max != max_component_size:
                compute_nx = True # NOTE re-using this flag is a bit adhoc, but it works
                min_component_size = new_min
                max_component_size = new_max

    self.max_comp_size = max_component_size
    self.min_comp_size = min_component_size
    diff = float(self.max_comp_size - self.min_comp_size)
    # Recompute normalization
    if compute_nx:
        # if there's more than one component, compute the normalized values
        if len(connected_components) > 1 and diff > 0:
            # Calculate (mycomponent-min_component) / (max_component-min_component)
            return (self.node_component_sizes - self.min_comp_size) / diff
        else: # otherwise, everyone is in the same sized component
            return np.ones(len(self.node_to_row.keys()))
    else:
        if diff > 0:
            new_val = float(self.connected_component_sizes[probed_comp]-self.min_comp_size)/diff
        else:
            new_val = 1.0
        current_norm_vals[np.where(self.components == probed_comp)] = new_val
        return current_norm_vals



def get_cc(self, node):
    """
    Takes a node as input and returns its clustering coefficient.
    NOTE: ASSUMES UNDIRECTED NETWORK
    """
    # get degree
    neighbors = self.sample_adjlist_sets[node]
    deg = len(neighbors)
    if deg > 1:
        denom = deg * (deg-1)
        num = 0
        # for each pair of neighbors
        for n1 in neighbors:
            n1_ne = self.sample_adjlist_sets[n1]
            for n2 in neighbors:
                if n1 == n2: # continue the loop if the nodes are the same
                    continue
                # if there is an edge between them
                if n1 in self.sample_adjlist_sets[n2] or n2 in n1_ne:
                    num += 1 # increment the number of edges
        return (float(num)) / float(denom)
    else:
        return 0.0

