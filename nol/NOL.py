#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import Network
import sys
import os
import logging


def RunEpisode(G, alpha, theta, epochs, Resultfile='output_file.txt', policy='NOL', regularization='nonnegative', featureOrder='linear',
             reward_function='new_nodes', saveGap=0, episode=0, iteration=0, p = None, decay=0, k=4, target_attribute = None):
    features = G.calculate_features(G, featureOrder)
    values = features.dot(theta)
    ## TODO Adhoc
    eligibilityTraces = np.zeros([len(theta), ])
    probedNodes = []
    unprobedNodeSet = G.sample_node_set.copy()
    unprobedNodeIndices = {G.node_to_row[i] for i in unprobedNodeSet}
    graphSaveInterval = saveGap

    targetNodeSet=set()
    if reward_function == 'attribute':
        targetNodeSet = {node for node in G.node_to_row \
                         if target_attribute in G.attribute_dict[node] and (len(G.complete_graph_adjlist[node]) == len(G.sample_graph_adjlist[node]))}
        initialTargetNodes = len(targetNodeSet)
        logging.info('# initial target nodes: ' + str(initialTargetNodes))
        for node in targetNodeSet:
            unprobedNodeSet.remove(node)
            unprobedNodeIndices.remove(G.node_to_row[node])
            G.probedNodeSet.add(node)


    intermediate_result_dir = os.path.join(Resultfile, 'intermediate_results')
    if not os.path.exists(intermediate_result_dir):
        os.makedirs(intermediate_result_dir)
    intermediate_name = os.path.join(intermediate_result_dir, policy + '_iter' + str(iteration) +
                                     '_a' + str(alpha) + '_episode_' + str(episode) +
                                     '_intermediate.txt')
    intermediateFile = open(intermediate_name, 'w+')
    intermediateFile.write('Epoch left' + '\t' + 'G_t' + '\t'+ 'V_t'+ '\t'+ 'Error = Actual-prediction' + '\t' + 'RegressionError' + '\t' + 'Rewards in each step' + '\t' + 'jump' + '\t' + 'value_MSE' + '\t' + 'theta_diff')
    for i in range(theta.shape[0]):
        intermediateFile.write('\tTheta[' + str(i) + ']')
    intermediateFile.write('\n')
    intermediate_graph_dir = os.path.join(Resultfile, 'intermediate_graphs')
    if not os.path.exists(intermediate_graph_dir):
        os.makedirs(intermediate_graph_dir)
    intermediateGraphFile = os.path.join(intermediate_graph_dir, policy +
                                                '_iter' + str(iteration) +
                                                '_a' + str(alpha) + '_episode_' + str(episode) +
                                                '_intermediate_graph_')
    featureFileDir = os.path.join(Resultfile, 'feature_analysis')
    if not os.path.exists(featureFileDir):
        os.makedirs(featureFileDir)

    rewards = []

    if reward_function == 'attribute':
        rewards = [1]*initialTargetNodes

    count = 0
    interval = 500
    for epoch in range(epochs):
        if count == interval:
            count = 0
            logging.info('Iteration: ' + str(iteration) + ' Epoch: ' + str(epoch))
        count += 1

        ## probe a node, get the reward
        ## need the number of nodes before the probe
        numberOfNodes = len(G.node_to_row)
        if reward_function == 'attribute':
            numberOfTargetNodes = len(targetNodeSet) - initialTargetNodes
        ## Choose a node (index) to probe
        nodeIndex, jump = action(G, policy, values, unprobedNodeIndices, p)

        ## Get the node id
        probedNode = G.row_to_node[nodeIndex]
        ## add to the probed nodes
        probedNodes.append(probedNode)

        if reward_function == 'new_edges':
            new_neighbors = G.complete_graph_adjlist[probedNode]
            num_new_edges = len([ne for ne in new_neighbors if ne not in G.sample_adjlist_sets[probedNode]])
        elif reward_function == 'nodes_and_triangles':
            new_neighbors = G.complete_graph_adjlist[probedNode]
            new_edges = [ne for ne in new_neighbors if ne not in G.sample_adjlist_sets[probedNode]]
            num_new_triangles = 0
            for u in new_edges:
                    for v in new_edges:
                        if u == v:
                            continue
                        ## check if a triangle was closed
                        if v in G.sample_adjlist_sets.keys() and u in G.sample_adjlist_sets[v]:
                            num_new_triangles += 1
                        elif u in G.sample_adjlist_sets.keys() and v in G.sample_adjlist_sets[u]:
                            num_new_triangles += 1

        ## Actually probe the node
        new_nodes = G.probe(probedNode)  # probe the node and get the novel neighbors
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
        if reward_function == 'new_nodes':
            reward = len(G.node_to_row) - numberOfNodes
        elif reward_function == 'new_edges':
            reward = num_new_edges
        elif reward_function == 'nodes_and_triangles':
            reward = (len(G.node_to_row) - numberOfNodes) + num_new_triangles
        elif reward_function == 'attribute':
            reward = len(targetNodeSet) - initialTargetNodes - numberOfTargetNodes

        ## Update reward
        rewards.append(reward)

        if policy == 'NOL-HTR':
            if epoch == 0:
                samples_mat = update_samples_matrix(None, features, nodeIndex, reward, reward_function, targetNodeSet)
            else:
                samples_mat = update_samples_matrix(samples_mat, features, nodeIndex, reward, reward_function, targetNodeSet)

        ## Need the features of the current sample
        features = G.F

        ## what is the value function of current state with current theta
        currentValue = values[nodeIndex]

        ## get the current gradient
        currentGradient = features[nodeIndex, :].copy()

        ## update the features    
        features = G.update_features(G, probedNode, order=featureOrder)

        ## compute value per node
        values = features.dot(theta)

        if len(unprobedNodeIndices) == 0:
            break

        old_theta = theta

        if policy == 'NOL':
            theta = online_regression_update(theta, alpha, reward, currentValue, currentGradient)
        elif policy == 'NOL-HTR':
            theta = median_of_means(samples_mat, theta, alpha, reward-currentValue, regularization, k, number_unprobed=len(unprobedNodeIndices))

        if regularization == 'nonnegative':
            theta[theta < 0] = 0
        elif regularization == 'normalized':
            theta = theta/np.linalg.norm(theta)
        elif regularization == 'pos_normalized':
            theta[theta < 0] = 0
            if np.linalg.norm(theta) > 0:
                theta = theta/np.linalg.norm( theta)
        elif regularization != 'no':
            logging.warning("Unrecognized regularization type\"" + regularization + "\\. Defaulting to no normalization.")

        theta_diff = sum([(theta[i]-old_theta[i])**2 for i in range(theta.shape[0])])

        ## Get the new value mapping
        old_values = values
        values = features.dot(theta)
        ## Get MSE between old and new values
        MSE = sum([(old_values[i]-values[i])**2 for i in range(old_values.shape[0])])

        if policy == 'NOL' or policy == 'globalmax_jump' or policy == 'globalmax_smartjump':
            ## print a 1 if a random node was chosen, a 0 if the model was followed
            if jump is True:
                jval = 1
            elif jump is False:
                jval = 0
            else:
                logging.info('SOMETHING IS WRONG WITH JUMP SWITCH!')

        try:
            # write intermediate numbers
            regret = 0
            intermediateFile.write(str(epoch) + '\t' + str(0) + '\t' + str(currentValue)  + '\t'+ str(regret) + '\t' +  str(0) + '\t' + str(reward) + '\t' + str(jump) + '\t' + str(MSE) + '\t' + str(theta_diff))


            for i in range(theta.shape[0]):
                intermediateFile.write('\t' + str(theta[i]))
            intermediateFile.write('\n')
        except Exception as e:
            print(e)

        graphSaveInterval += 1

        if len(unprobedNodeIndices) == 0:
            break

    logging.info('Total reward: ' + str(sum(rewards)))
    print(str(sum(rewards)))
    intermediateFile.close()

    return probedNodes, theta, rewards


def action(G, policy, values, unprobedNodeIndices, p = -1):
    """
    Accepts: Network object, list of locally adjacent nodes to the most
                recently probed node (for local methods).
                The policy to probe with.
                Current value function.
                Set of unprobed indices.
    Returns: the next action (node index to probe) dictated by the policy.
    """
    idx = []
    unprobedNodeList = [row for row in G.row_to_node.keys() if row in unprobedNodeIndices]
    if policy == 'globalmax_jump':
        prob = np.random.random()
        ## With probability 1-p, follow global max
        if prob > p:
            idx = np.argmax(values[unprobedNodeList])
            return unprobedNodeList[idx], False
        else:
            ## with probability p, pick a node at random
            return np.random.choice(unprobedNodeList, 1)[0], True
    elif policy == 'NOL' or policy == 'NOL-HTR':
        prob = np.random.random()
        ## With probability 1-p, follow global max
        if prob > p:
            idx = np.argmax(values[unprobedNodeList])
            return unprobedNodeList[idx], False
        else:
            ## with probabilit p, pick a node at random
            ## Until there are none left, pick nodes from
            # the initial sample
            not_probed_initial = list(G.original_node_set - G.probedNodeSet)
            if len(not_probed_initial) > 0:
                node = np.random.choice(not_probed_initial, 1)[0]
                return G.node_to_row[node], True
            else:
                return np.random.choice(unprobedNodeList, 1)[0], True
    else:
        logging.warning("Unrecognized policy \"" + policy + "\". Exiting.")
        sys.exit(1)


########################### CODE FOR NOL ONLINE REGRESSION ###########################

def online_regression_update(theta, alpha, reward, value, node_features):
    loss = (reward - value)**2

    gradient = -2*(reward - value) * node_features

    ## update theta
    theta = theta + alpha*gradient.T

    return theta


########################### CODE FOR NOL-HTR ###########################

def median_of_means(samples_mat, theta, alpha, delta, regularization, k_input, confidence=0.05, lambda_regres=0.0, number_unprobed=0):

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
            ## TODO Add lambda_regres in here
            vals = [(thetas[i] - thetas[j]).dot((covariances[j])*(thetas[i] - thetas[j]))\
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


###### CODE FOR STARTING EXPERIMENT ######
def RunIteration(G, alpha_input, episodes, epochs , initialNodes, Resultfile='output_file.txt', policy='NOL', regularization = 'nonnegative', order =
                 'linear', reward_function = 'new_nodes', saveGAP = 0, current_iteration=0, p = None, decay=0, k=4, target_attribute = None):
    theta_estimates = np.random.uniform(-0.2, 0.2,(G.get_numfeature(),))     # Initialize estimates at all 0.5
    initial_graph = G.copy()

    for episode in range(episodes):
        logging.info("episode: " + str(episode))
        probed_nodes, theta, rewards = RunEpisode(G, alpha_input, theta_estimates, epochs, Resultfile, policy, regularization, order, reward_function, saveGAP, episode, current_iteration, p, decay, k, target_attribute)

        theta_estimates = theta # Update value estimates
        G = initial_graph.copy() # reset G to the original sample
    return probed_nodes, theta, rewards
