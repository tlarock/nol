#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import Network
import sys
import os
import logging
from puAdapter import *
from sklearn_classifiers import *



def nol(G, alpha, budget, output_dir='output_file.txt', policy='NOL', regularization='nonnegative',
             reward_function='new_nodes', saveGap=0, iteration=0, epsilon = None, decay=0, k=4, target_attribute = None, burn_in=0):
    '''
    Beginning from a sample network, run an algorithm that grows the network by querying nodes and
    acquiring their information.

    Parameters
    ----------
    G (nol.Network)
        A Network object. This object contains both the sampled network and the complete network.
    alpha (float)
        Learning rate for the learning algorithms
    theta (np.array)
        Parameters to be learned, usually initialized randomly
    budget (int)
        Number of queries to make of the underlying network. The budget for probing.
    output_dir (str)
        Path to an output directory
    policy (str)
        The method for choosing which node to query. Can require learning (e.g. NOL, NOL-HTR) or heuristic (e.g. high, rand).
    regularization (str)
        Method for regularizing parameters theta.
    reward_function (str)
        Objective to be maximized. Defaul it new_nodes (maximize node cover)
    saveGap (int)
        If > 0, features and graph will be output every saveGap queries. if <=0, graph/node data is not saved.
    iteration (int)
        Iteration number. Used to construct filename.
    epsilon (float)
        Jump probability for epsilon-greedy methods
    decay (int)
        If 1, apply exponential decay over time to jump probability p
    k (int)
        k for Heavy Tail Regression
    target_attribute (int)
        Index in attribute_dict[node] corresponding to the attribute of interest if using NOL for search.
    burn_in (int)
        Number of queries to conduct using heuristic method to train parameters.

    Returns
    -------
    probedNodes (list)
        List of nodes that were probed in the experiment
    theta (np.array)
        Parameters learned through querying (if applicable; otherwise random vector)
    rewards (list)
        List of rewards for each query.
    '''

    if policy not in ['high', 'low', 'rand']:
        features = G.calculate_features(G)
        theta = np.random.uniform(-0.2, 0.2, (G.get_numfeature(),))
        samples_mat = None
    else:
        features = None
        samples_mat = None
        theta = None

    if decay == 1:
        input_epsilon = epsilon

    probedNodes = []
    unprobedNodeSet = G.sample_node_set.copy()
    unprobedNodeIndices = {G.node_to_row[i] for i in unprobedNodeSet}
    graphSaveInterval = saveGap

    ## initialize log files
    intermediate_result_dir = os.path.join(output_dir, 'intermediate_results')
    if not os.path.exists(intermediate_result_dir):
        os.makedirs(intermediate_result_dir)
    intermediate_name = os.path.join(intermediate_result_dir, policy + '_iter' + str(iteration) + '_intermediate.txt')
    with open(intermediate_name, 'w') as intermediateFile:
        intermediateFile.write('query\treward\testimate\tdelta\tjump\tp')
        if policy not in ['high', 'low', 'rand']:
            for i in range(theta.shape[0]):
                intermediateFile.write('\tTheta[' + str(i) + ']')
        intermediateFile.write('\n')

    intermediate_graph_dir = os.path.join(output_dir, 'intermediate_graphs')
    if not os.path.exists(intermediate_graph_dir):
        os.makedirs(intermediate_graph_dir)
    intermediateGraphFile = os.path.join(intermediate_graph_dir, policy + '_iter' + str(iteration) + '_graph.txt')
    open(intermediateGraphFile, 'w').close()
    if policy not in ['high', 'low', 'rand']:
        featureFileDir = os.path.join(output_dir, 'feature_analysis')
        if not os.path.exists(featureFileDir):
            os.makedirs(featureFileDir)

    ## If doing attribute search, initialize data structure
    targetNodeSet=set()
    if reward_function == 'attribute':
        ## assumes 'netdisc' sampling strategy
        targetNodeSet = {node for node in G.node_to_row \
                         if target_attribute in G.attribute_dict[node] and (len(G.complete_graph_adjlist[node]) == len(G.sample_graph_adjlist[node]))}
        initialTargetNodes = len(targetNodeSet)
        logging.info('# initial target nodes: ' + str(initialTargetNodes))
        i = 0
        for node in targetNodeSet:
            unprobedNodeSet.remove(node)
            unprobedNodeIndices.remove(G.node_to_row[node])
            G.probedNodeSet.add(node)
            if i == 0:
                samples_mat = np.append(features[G.node_to_row[node]], np.array([1]))
                if len(samples_mat.shape) == 1:
                    ## If we use 1 seed, reshape the array to work with classifiers
                    samples_mat = samples_mat.reshape(1, samples_mat.shape[0])
            else:
                samples_mat = np.vstack( (samples_mat, np.append(features[G.node_to_row[node]], np.array([1])) ) )
            i+=1

    rewards = []
    if reward_function == 'attribute':
        rewards = [1]*initialTargetNodes

    count = 0
    print_interval = 500
    query = 0

    ## Default: no burn-in phase
    if burn_in <= 0:
        if policy in ['NOL-HTR', 'NOL']:
            values = get_values(G, 'NOL', samples_mat, features, unprobedNodeIndices, unprobedNodeSet, theta)
        else:
            values = get_values(G, policy, samples_mat, features, unprobedNodeIndices, unprobedNodeSet, theta)

    elif burn_in > 0:
        ## burn in phase
        logging.info('# burn in queries: ' + str(burn_in))
        for _ in range(burn_in):
            if reward_function == 'attribute':
                numberOfTargetNodes = len(targetNodeSet) - initialTargetNodes

            deg_values = {node:len(G.sample_graph_adjlist[G.row_to_node[node]]) for node in unprobedNodeIndices}
            nodeIndex = max(deg_values.items(), key=lambda kv: kv[1])[0]
            probedNode = G.row_to_node[nodeIndex]
            new_nodes = G.probe(probedNode)
            if reward_function == 'attribute':
                if target_attribute in G.attribute_dict[probedNode]:
                    targetNodeSet.add(probedNode)

            new_nodes = set(new_nodes)
            new_unprobed = new_nodes - G.probedNodeSet

            ## Update the sets of nodes 
            unprobedNodeSet.update(new_unprobed)
            assert probedNode in unprobedNodeSet, 'probedNode ' + str(probedNode) + ' not in unprobednodeset!'

            unprobedNodeSet.remove(probedNode)
            new_unprobed_indices = {G.node_to_row[node] for node in new_unprobed}
            unprobedNodeIndices.update(new_unprobed_indices)
            unprobedNodeIndices.remove(nodeIndex)
            assert unprobedNodeSet.isdisjoint(G.probedNodeSet)

            ## Calculate absolute reward
            if reward_function == 'attribute':
                absoluteReward = len(targetNodeSet) - initialTargetNodes - numberOfTargetNodes

            ## Update reward
            rewards.append(absoluteReward)

            ## update samples matrix
            if policy not in ['high', 'low', 'rand']:
                samples_mat = np.vstack( (samples_mat, np.append(features[nodeIndex], np.array([absoluteReward])) ) )
                features = G.update_features(G, probedNode)

            values = get_values(G, policy, samples_mat, features, unprobedNodeIndices, unprobedNodeSet, theta)

            write_intermediate(query, absoluteReward, 0, 0, 0, p, theta, intermediate_name)
            write_query(G, probedNode, targetNodeSet, intermediateGraphFile)
            if (saveGap != 0 and graphSaveInterval == (saveGap)) or query == (budget-1):
                if query == (budget - 1):
                    query += 1
                graphSaveInterval = 0
                if policy not in ['high', 'low', 'rand']:
                    featureFileName = policy + '_iter' + str(iteration) + '_features_' + str(query) + '.txt'
                    featureFile = os.path.abspath(os.path.join(featureFileDir, featureFileName))
                    write_features(G, features, featureFile)

            graphSaveInterval += 1
            query += 1

    ## Begin querying following specified policy
    while query < budget:
        if count == print_interval:
            count = 0
            logging.info('Iteration: ' + str(iteration) + ' Epoch: ' + str(query))
        count += 1

        ## probe a node, get the reward
        ## need the number of nodes before the probe
        numberOfNodes = len(G.node_to_row)
        if reward_function == 'attribute':
            numberOfTargetNodes = len(targetNodeSet) - initialTargetNodes 

        ## Choose a node (index) to probe
        nodeIndex, jump = action(G, policy, values, unprobedNodeIndices, epsilon)

        ## Get the node id
        probedNode = G.row_to_node[nodeIndex]
        ## add to the probed nodes
        probedNodes.append(probedNode)

        if reward_function == 'new_edges':
            new_neighbors = G.complete_graph_adjlist[probedNode]
            num_new_edges = len([ne for ne in new_neighbors if ne not in G.sample_adjlist_sets[probedNode]])

        ## Actually probe the node
        new_nodes = G.probe(probedNode)  # probe the node and get the novel neighbors
        new_nodes = set(new_nodes)
        if reward_function == 'attribute':
            if target_attribute in G.attribute_dict[probedNode]:
                targetNodeSet.add(probedNode)

        new_unprobed = new_nodes - G.probedNodeSet

        ## Update the sets of nodes 
        unprobedNodeSet.update(new_unprobed)

        assert probedNode in unprobedNodeSet, 'probedNode ' + str(probedNode) + ' not in unprobednodeset!'

        unprobedNodeSet.remove(probedNode)
        new_unprobed_indices = {G.node_to_row[node] for node in new_unprobed}
        unprobedNodeIndices.update(new_unprobed_indices)
        unprobedNodeIndices.remove(nodeIndex)

        assert unprobedNodeSet.isdisjoint(G.probedNodeSet)

        ## Calculate absolute reward
        if reward_function == 'new_nodes':
            reward = len(G.node_to_row) - numberOfNodes
        elif reward_function == 'new_edges':
            reward = num_new_edges
        elif reward_function == 'attribute':
            reward = len(targetNodeSet) - initialTargetNodes - numberOfTargetNodes

        ## Update reward
        rewards.append(reward)

        if policy == 'NOL-HTR':
            if query == 0:
                samples_mat = update_samples_matrix(None, features, nodeIndex, reward, reward_function, targetNodeSet)
            else:
                samples_mat = update_samples_matrix(samples_mat, features, nodeIndex, reward, reward_function, targetNodeSet)

        if policy not in ['high', 'low', 'rand']:
            ## Need the features of the current sample
            features = G.F

            ## what is the value function of current state with current theta
            currentValue = values[nodeIndex]

            ## get the current gradient
            currentGradient = features[nodeIndex, :].copy()

            ## update the features    
            features = G.update_features(G, probedNode)

            ## Delta is the difference from between expecation and reward
            delta = reward - currentValue

        if len(unprobedNodeIndices) == 0:
            break

        old_theta = theta

        if policy == 'NOL':
            theta = online_regression_update(theta, alpha, delta, currentGradient)
        elif policy == 'NOL-HTR':
            theta = median_of_means(samples_mat, theta, alpha, reward-currentValue, regularization, k, number_unprobed=len(unprobedNodeIndices))

        ## regularize the parameters
        theta = regularize_theta(theta, regularization)

        ## Get the new value mapping
        ## compute value per node
        values = get_values(G, policy, samples_mat, features, unprobedNodeIndices, unprobedNodeSet, theta)

        ## decay epsilon if necessary
        if decay == 1:
            epsilon = input_epsilon * (np.exp(-1*query/budget))

        if len(unprobedNodeIndices) == 0:
            break

        if policy not in ['high', 'low', 'rand']:
            write_intermediate(query, reward, currentValue, delta, jump, epsilon, theta, intermediate_name)
        else:
            write_intermediate(query, reward, 0, 0, jump, epsilon, None, intermediate_name)

        ## Write graph query to output file
        write_query(G, probedNode, targetNodeSet, intermediateGraphFile)
        if (saveGap != 0 and graphSaveInterval == (saveGap)) or (query == (budget-1) and saveGap > 0):
            graphSaveInterval = 0
            if policy not in ['high', 'low', 'rand']:
                featureFileName = policy + '_iter' + str(iteration) + '_features_' + str(query) + '.txt'
                featureFile = os.path.abspath(os.path.join(featureFileDir, featureFileName))
                write_features(G, features, featureFile)

        graphSaveInterval += 1
        query += 1

    logging.info('Total reward: ' + str(sum(rewards)))
    intermediateFile.close()

    return probedNodes, theta, rewards


def action(G, policy, values, unprobedNodeIndices, epsilon = -1):
    """
    Parameters
    ----------
    G: (nol.Network)
        The observed network object
    policy: (str)
        The policy to choose the next node to query
    values: (dict)
        Current value function, form <node_id, predicted_val>
    unprobedNodeIndices: (list)
        List of indices of unprobed nodes
    epsilon: (float)
        Jump probability.

    Returns
    -------
    idx: (int)
        The node index of the next action
    jump: (int)
        If 1, then action is a random jump. If 0, action is following policy.

    """
    idx = []
    unprobedNodeList = [row for row in G.row_to_node.keys() if row in unprobedNodeIndices]

    prob = np.random.random()
    if policy == 'rand':
        ## Pick a uniformly random node
        idx = np.random.choice(unprobedNodeList, 1)[0]
        return idx, 1
    elif prob > epsilon:
        ## NOTE: Default value of epsilon falls here
        if policy == 'low':
            idx = min(values.items(), key=lambda kv: kv[1])[0]
        else:
            ## Choose the node with the maximum value
            idx = max(values.items(), key=lambda kv: kv[1])[0]
        return idx, 0
    else:
        ## with probability epsilon, pick a node at random
        ## choose all of the initial sample nodes first,
        ## then uniform from all remaining unprobed
        not_probed_initial = list(G.original_node_set - G.probedNodeSet)
        if len(not_probed_initial) > 0:
            node = np.random.choice(not_probed_initial, 1)[0]
            return G.node_to_row[node], 1
        else:
            return np.random.choice(unprobedNodeList, 1)[0], 1

def get_values(G, policy, samples_mat, features, unprobedNodeIndices, unprobedNodeSet, theta=None):
    '''
    Compute and return the value function for a given policy.
    '''
    if policy in ['NOL', 'NOL-HTR']:
        values = features.dot(theta)
        values = {idx:values[idx] for idx in G.row_to_node.keys() if idx in unprobedNodeIndices}
    elif policy == 'svm':
        values = compute_svm_values(samples_mat, features, unprobedNodeIndices)
    elif policy == 'knn':
        values = compute_knn_values(samples_mat, features, unprobedNodeIndices)
    elif policy == 'logit':
        y = samples_mat[:,samples_mat.shape[1]-1]
        ## if there is only 1 class, use the 1 class prediction
        if np.unique(y).shape[0] == 1:
            for node in unprobedNodeSet:
                all_unprobed_mat = np.array(samples_mat)
                all_unprobed_mat = np.vstack( (all_unprobed_mat, np.append(features[G.node_to_row[node]], np.array([-1]))))
            values, theta = compute_logit_values(all_unprobed_mat, features, unprobedNodeIndices, one_class=True)
        else:
            values, theta = compute_logit_values(samples_mat, features, unprobedNodeIndices)
    elif policy == 'linreg':
        values, theta = compute_linreg_values(samples_mat, features, unprobedNodeIndices)
    elif policy in ['high', 'low', 'rand']:
        values = compute_deg_values(G, unprobedNodeIndices)

    return values


def regularize_theta(theta, regularization):
    '''
    Regularize the theta parameter according to specified regularization.
    '''
    if regularization == 'nonnegative':
        theta[theta < 0] = 0
    elif regularization == 'normalized':
        theta = theta/np.linalg.norm(theta)
    elif regularization == 'pos_normalized':
        theta[theta < 0] = 0
        if np.linalg.norm(theta) > 0:
            theta = theta/np.linalg.norm(theta)
    elif regularization != 'no':
        logging.warning("Unrecognized regularization type\"" + regularization + "\\. Defaulting to no normalization.")

    return theta

########################### CODE FOR NOL ONLINE REGRESSION ###########################

def online_regression_update(theta, alpha, delta, node_features):
    '''
    Updates parameters using online linear regression (Strehl & Littman 2008)

    Parameters
    ----------
    theta: (np.array)
        the current parameters
    alpha: (float)
        learning rate
    reward: (float/int)
        actual reward from query
    value: (float/int)
        predicted reward before query
    node_feautres: (np.array)
        the features of the queried node

    Returns
    -------
    theta: (np.array)
        the updated parameters
    '''
    ## Compute the gradient (based on loss=(reward-value)**2)
    gradient = (2*(delta)) * node_features

    ## update theta
    theta = theta + (alpha*gradient.T)

    return theta


########################### CODE FOR NOL-HTR ###########################

def median_of_means(samples_mat, theta, alpha, delta, regularization, k_input, confidence=0.05, lambda_regres=0.0, number_unprobed=0):
    '''
    Computes parameters using heavy tail regression (TODO: Citation)

    Parameters
    ----------
    samples_mat: (np.array)
    '''

    n = samples_mat.shape[0]
    if isinstance(k_input, int):
        k = k_input
    elif isinstance(k_input, type(np.log2)):
        k = int(k_input(n))
        if k < 1:
            k = 1
    else:
        print('Invalid value of k_input: ' + str(k_input))
        raise Exception

    if n/k < 1:
        ## Regress separately on every sample
        k = n

    features = samples_mat[:,0:samples_mat.shape[1]-1]
    ## TODO If the reward function is binary, should add some small response value for 0, otherewise the best parameters will be 0.
    y = samples_mat[:,samples_mat.shape[1]-1]
    ## Implement median of means
    if n/k > 2000:
        num_samples = 2000
    else:
        num_samples = int(n/k)

    samples_idx_list = np.random.choice(list(range(n)), (k, num_samples), replace=False)
    covariances = []
    thetas = []
    for i in range(k):
        ## get features and response for S_i
        idx_list = samples_idx_list[i]
        curr_features = features[idx_list,:]
        curr_y = y[idx_list]
        ## compute the paramters for this subsample based on pseudo-inverse (Moore-Penrose) MLE
        curr_theta = np.linalg.pinv(curr_features.T.dot(curr_features)).dot(curr_features.T.dot(curr_y))
        ## Compute covariance for each subsample
        covariances.append(np.cov(curr_features, rowvar=False))

        thetas.append(curr_theta)

    ## Compute medians of means
    medians = []
    if n > 1:
        for i in range(k):
            ## TODO This implementation assumes lambda = 0
            vals = [(thetas[i] - thetas[j]).dot((covariances[j])*(thetas[i] + thetas[j]))\
                    for j in range(k) if j != i]
            medians.append(np.median(vals))

        ## return the parameters associated with the minimum median
        new_theta = thetas[np.argmin(medians)]
    else:
        new_theta = thetas[0]

    return new_theta


def update_samples_matrix(samples_mat, features, nodeIndex, reward, reward_function, targetNodeSet):
    ## Update sampled matrix
    if samples_mat is not None:
        samples_mat = np.vstack( (samples_mat, np.append(features[nodeIndex], np.array([reward])) ) )
    else:
        samples_mat = np.append(features[nodeIndex], np.array([reward]))
        samples_mat = np.reshape(samples_mat, (1, samples_mat.shape[0]))
        if reward_function == 'attribute':
            for node in targetNodeSet:
                samples_mat = np.vstack( (samples_mat, np.append(features[G.node_to_row[node]], np.array([1])) ) )

    return samples_mat


###### CODE FOR WRITING TO FILES    ######
def write_query(G, probed_node, targetNodeSet, intermediateGraphFile):
    with open(intermediateGraphFile, 'a') as f_graph:
        if probed_node in targetNodeSet:
            label = '1'
        else:
            label = '0'
        f_graph.write(str(probed_node) + ':' + label)

        for u in G.sample_graph_adjlist[probed_node].keys():
            if u in G.probedNodeSet and u in targetNodeSet:
                u_lab = '1'
            elif u in G.probedNodeSet:
                u_lab = '0'
            else:
                u_lab = '-1'
            f_graph.write(',' + str(u) + ':' + u_lab)

        f_graph.write('\n')

def write_features(G, features, featureFile):
    with open(featureFile, 'w') as f_feats:
        for i in range(features.shape[0]):
            f_feats.write(str(G.row_to_node[i]) + ',' + ','.join([str(val) for val in features[i]]) + '\n')


def write_intermediate(query, reward, currentValue, delta, jump, p, theta, intermediate_name):
    with open(intermediate_name, 'a') as intermediateFile:
        # write intermediate numbers
        intermediateFile.write(str(query) + '\t' + str(reward) +  '\t' + str(currentValue)  + '\t'+ str(delta) + '\t' + str(jump) + '\t' + str(p))

        if theta is not None:
            for i in range(theta.shape[0]):
                intermediateFile.write('\t' + str(theta[i]))
        intermediateFile.write('\n')
