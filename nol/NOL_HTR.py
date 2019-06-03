#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from sklearn import linear_model
import utility
import Network
import sys
import os
import logging
from scipy import stats

def RunEpisode(G, alpha, lambda_, gamma, theta, epochs, Resultfile='output_file.txt',
             updateType='qlearning', policy='random', regularization='nonnegative', featureOrder='linear',
             reward_function='new_nodes', saveGap=0, episode=0, iteration=0, p = None, decay=0, k=4, target_attribute = None):
    print(policy)
    features = G.calculate_features(G, featureOrder)
    values = features.dot(theta)
    ## TODO Adhoc
    values_list = []
    rewards_list = []
    eligibilityTraces = np.zeros([len(theta), ])
    probedNodes = []
    unprobedNodeSet = G.sample_node_set.copy()
    unprobedNodeIndices = {G.node_to_row[i] for i in unprobedNodeSet}
    graphSaveInterval = saveGap

    if reward_function == 'attribute':
        targetNodeSet = set()
        for node in G.node_to_row:
            if target_attribute in G.attribute_dict[node]:
                if len(G.complete_graph_adjlist[node]) == len(G.sample_graph_adjlist[node]):
                    print(node)
                    targetNodeSet.add(node)
                    unprobedNodeSet.remove(node)
                    unprobedNodeIndices.remove(G.node_to_row[node])

        initialTargetNodes = len(targetNodeSet)
        logging.info('# initial target nodes: ' + str(initialTargetNodes))

    intermediate_result_dir = os.path.join(Resultfile, 'intermediate_results')
    if not os.path.exists(intermediate_result_dir):
        os.makedirs(intermediate_result_dir)
    intermediate_name = os.path.join(intermediate_result_dir, 'LTD_'+ policy + '_iter' + str(iteration) +
                                     '_a' + str(alpha) + '_l' + str(lambda_) +
                                     '_g' + str(gamma) + '_episode_' + str(episode) +
                                     '_intermediate.txt')
    intermediateFile = open(intermediate_name, 'w+')
    if policy == 'globalmax_adaptive':
        intermediateFile.write('Epoch left' + '\t' + 'G_t' + '\t'+ 'V_t'+ '\t'+ 'Error = Actual-prediction' + '\t' + 'RegressionError' + '\t' + 'Rewards in each step' + '\t' + 'jump' + '\t' + 'p' + '\t' + 'f' + '\t' + 'delta' + '\t' + 'value_MSE')
    else:
        intermediateFile.write('Epoch left' + '\t' + 'G_t' + '\t'+ 'V_t'+ '\t'+ 'Error = Actual-prediction' + '\t' + 'RegressionError' + '\t' + 'Rewards in each step' + '\t' + 'jump' + '\t' + 'value_MSE' + '\t' + 'theta_diff')
    for i in range(theta.shape[0]):
        intermediateFile.write('\tTheta[' + str(i) + ']')
    intermediateFile.write('\n')
    intermediate_graph_dir = os.path.join(Resultfile, 'intermediate_graphs')
    if not os.path.exists(intermediate_graph_dir):
        os.makedirs(intermediate_graph_dir)
    intermediateGraphFile = os.path.join(intermediate_graph_dir, 'LTD_' + policy +
                                                '_iter' + str(iteration) +
                                                '_a' + str(alpha) + '_l' + str(lambda_) +
                                                '_g' + str(gamma) + '_episode_' + str(episode) +
                                                '_intermediate_graph_')
    featureFileDir = os.path.join(Resultfile, 'feature_analysis')
    if not os.path.exists(featureFileDir):
        os.makedirs(featureFileDir)


    rewards = []

    if reward_function == 'attribute':
        rewards = [1]*initialTargetNodes

    count = 0
    interval = 500
    epoch = 0
    while epoch < epochs:
        values_list.append(values[list(unprobedNodeIndices)])
        rewards_list.append([(len(G.complete_graph_adjlist[node]) - len(G.sample_adjlist_sets[node])) for node in unprobedNodeSet])


        # print for logging purposes...
        if count == interval:
            count = 0
            logging.info("Iteration: " + str(iteration) + " Epoch: " + str(epoch))
        count += 1

        ## probe a node, get the reward
        ## need the number of nodes before the probe
        numberOfNodes = len(G.node_to_row)
        if reward_function == 'attribute':
            numberOfTargetNodes = len(targetNodeSet) - initialTargetNodes

        ## need the last probed node
        lastProbed = None
        if len(probedNodes) > 0:
            lastProbed = probedNodes[-1]
        else:
            lastProbed = np.random.choice(list(G.node_to_row.keys()), 1)[0]

        ## Always pass the neighbors of the previously probed node
        adjacentNodes = G.get_adjlist(lastProbed)

        ## Choose a node to probe
        nodeIndex, jump = action(G, adjacentNodes, policy, values, unprobedNodeIndices, p)

        ## find the index node to probe according to policy
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
            absoluteReward = len(G.node_to_row) - numberOfNodes
        elif reward_function == 'new_nodes_local':
            absoluteReward = len(G.node_to_row) - numberOfNodes
            #if G.rewardCounts[probedNode] > 0:
            #    absoluteReward += np.log2(G.rewardCounts[probedNode])
            absoluteReward += G.rewardCounts[probedNode]

        elif reward_function == 'new_edges':
            absoluteReward = num_new_edges
        elif reward_function == 'nodes_and_triangles':
            absoluteReward = (len(G.node_to_row) - numberOfNodes) + num_new_triangles
        elif reward_function == 'attribute':
            absoluteReward = len(targetNodeSet) - initialTargetNodes - numberOfTargetNodes

        reward = absoluteReward

        ## Update reward
        rewards.append(absoluteReward)

        ## TODO Update sampled matrix
        if epoch == 0:
            samples_mat = np.append(features[nodeIndex], np.array([absoluteReward]) )
            samples_mat = np.reshape(samples_mat, (1, samples_mat.shape[0]))
            if reward_function == 'attribute':
                for node in targetNodeSet:
                    samples_mat = np.vstack( (samples_mat, np.append(features[G.node_to_row[node]], np.array([1])) ) )
        else:
            samples_mat = np.vstack( (samples_mat, np.append(features[nodeIndex], np.array([absoluteReward])) ) )

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

        ## get value estimate on the next step given current theta
        adjacentNodeIndex = G.get_adjlist(probedNode)

        if updateType is 'sarsa':
            ## TODO ad-hoc!
            if policy == 'globalmax_jump' or policy == 'globalmax_restart' or policy == 'globalmax_smartjump' or policy == 'globalmax_adaptive' or policy == 'MoMs':
                next_node, _ = action(G, adjacentNodeIndex, policy, values, unprobedNodeIndices, p)
            else:
                next_node = action(G, adjacentNodeIndex, policy, values, unprobedNodeIndices, p)
            nextValue = values[next_node]
        elif updateType is 'qlearning':
            nextValue = values[action(G, adjacentNodeIndex, 'max', values, unprobedNodeIndices, p)]
        else:
            logging.warning("Unrecognized update type \"" + updateType + "\"")
            sys.exit(1)

        ## Calculate the estimated future reward
        estimatedReward = reward + gamma * nextValue
        delta = estimatedReward - currentValue

        ## Update the eligibility trace
        eligibilityTraces = lambda_ * gamma * eligibilityTraces + currentGradient.T

        ## update theta
        old_theta = theta
        theta = median_of_means(samples_mat, theta, alpha, delta, eligibilityTraces, regularization, k, number_unprobed=len(unprobedNodeIndices))
        theta_diff = sum([(theta[i]-old_theta[i])**2 for i in range(theta.shape[0])])

        ## Get the new value mapping
        old_values = values
        values = features.dot(theta)
        ## Get MSE between old and new values
        MSE = sum([(old_values[i]-values[i])**2 for i in range(old_values.shape[0])])
        if policy == 'globalmax_adaptive':
            adaptive_delta = 1.0 / len(unprobedNodeIndices)
            #std_alpha = 0.0
            #pos_values = values[np.where(values>0)]
            #if len(pos_values[pos_values> (pos_values.mean() + std_alpha*pos_values.std())]) > 0:
            #    adaptive_delta = 1.0 / len(pos_values[pos_values> (pos_values.mean() + std_alpha*pos_values.std())])
            #else:
            #    ## default to 1 / #actions
            #    adaptive_delta = 1.0 / len(unprobedNodeIndices)

            sigma = 0.1
            p,f = update_p(G, adjacentNodeIndex, values, unprobedNodeIndices, p, delta, alpha, sigma, adaptive_delta)

        ## TODO ad-hoc p decay
        if decay == 1:
            if epoch == 0:
                original_p = p
            p = original_p * (np.exp(-1*epoch/epochs))

        # writting intermediate numbers
        if policy == 'globalmax_restart' or policy == 'globalmax_jump' or policy == 'globalmax_smartjump':
            ## print a 1 if a random node was chosen, a 0 if the model was followed
            if jump is True:
                jval = 1
            elif jump is False:
                jval = 0
            else:
                logging.info('SOMETHING IS WRONG WITH JUMP SWITCH!')

        try:
            # write intermediate numbers
            if policy == 'globalmax_adaptive':
                intermediateFile.write(str(epoch) + '\t' + str(estimatedReward) + '\t' +
                    str(currentValue)  + '\t'+ str(delta) + '\t' +  str(delta) + '\t' + str(reward)
                    + '\t' + str(jump) + '\t' + str(p) + '\t' + str(f) + '\t' + str(adaptive_delta) + '\t' + str(MSE))
            else:
                intermediateFile.write(str(epoch) + '\t' + str(estimatedReward) + '\t' + str(currentValue)  + '\t'+ str(delta) + '\t' +  str(delta) +
                                       '\t' + str(reward) + '\t' + str(jump) + '\t' + str(MSE) + '\t' + str(theta_diff))

            for i in range(theta.shape[0]):
                intermediateFile.write('\t' + str(theta[i]))
            intermediateFile.write('\n')
        except Exception as e:
            print(e)

        if (saveGap != 0 and graphSaveInterval == (saveGap)) or epoch == (epochs-1):
            if epoch == (epochs - 1):
                epoch += 1
            graphSaveInterval = 0
            Ftemp = open(intermediateGraphFile + str(epoch) + '.txt', 'w')
            for key, val in G.sample_graph_adjlist.items():
                for key2, val2 in val.items():
                    Ftemp.write(str(key) + ' ' + str(key2) + '\n')
            Ftemp.close()
            # Snapshot of feature matrix, to be used with SummarizeFeatures.py
            #featureFileDir = '../results/feature_analysis/'
            featureFileName = 'FeaturesLTD_' + str(policy) + '_iter' + str(iteration) +\
                              '_a' + str(alpha) + '_l' + str(lambda_) + \
                              '_g' + str(gamma) + '_episode' + str(episode) +\
                              '_epoch' + str(epoch)
            featureFile = os.path.abspath(os.path.join(featureFileDir, featureFileName))
            np.savetxt(featureFile,features)

        graphSaveInterval += 1

        ## increment epoch
        epoch +=1

    intermediateFile.close()
    print(sum(rewards))
    ## TODO ad hoc
    reward_dist_file = 'reward_dist_' + str(policy) + '_iter' + str(iteration) +\
                          '_a' + str(alpha) + '_l' + str(lambda_) + \
                          '_g' + str(gamma) + '_episode' + str(episode) +\
                          '_epoch' + str(epoch)

    with open(os.path.abspath(os.path.join(featureFileDir, reward_dist_file)) + '.csv', 'w') as f:
        for i in range(len(rewards_list)):
            for j in range(len(rewards_list[i])):
                if j < len(rewards_list[i]) - 1:
                    f.write(str(rewards_list[i][j]) + ',')
                else:
                    f.write(str(rewards_list[i][j]) + '\n')



    logging.info('Total reward: ' + str(sum(rewards)))
    return probedNodes, theta, rewards


def update_p(G, adjnode, values, unprobedNodeIndices, p, td_error, alpha, sigma, delta):
    idx = []
    restart_probability = utility.getProbRestart()
    unprobedNodeList = [row for row in G.row_to_node.keys() if row in unprobedNodeIndices]
    f = (1 - np.exp( (-alpha*td_error) / sigma)) / (1 + np.exp( (-alpha*td_error) / sigma))
    p = delta * f + (1-delta)*p
    return p, f



def median_of_means(samples_mat, theta, alpha, delta, eligibilityTraces, regularization, k_input, confidence=0.05, lambda_regres=0.0, number_unprobed=0):

    n = samples_mat.shape[0]

    ## TODO Set k rather than choosing it (currenlty not using confidence param)
    #k = int(np.ceil(np.log2(number_unprobed)))
    #k = int(np.ceil(np.log2(1./confidence)))
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
    ## TODO If the reward function is binary, should add some small response value for 0, otherewise the best parameters will be 0...
    y = samples_mat[:,samples_mat.shape[1]-1] + 0.001
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

def action(G, adjnode, policy, values, unprobedNodeIndices, p = -1):
    """
    Accepts: Network object, list of locally adjacent nodes to the most
                recently probed node (for local methods).
                The policy to probe with.
                Current value function.
                Set of unprobed indices.
    Returns: the next action (node index to probe) dictated by the policy.
    """
    idx = []
    restart_probability = utility.getProbRestart()
    unprobedNodeList = [row for row in G.row_to_node.keys() if row in unprobedNodeIndices]
    if policy == 'globalmax':
        return unprobedNodeList[np.argmax(values[unprobedNodeList])]
    elif policy == 'globalmax_jump' or policy == 'MoMs':
        prob = np.random.random()
        ## With probability p, follow global max
        if prob > p:
            idx = np.argmax(values[unprobedNodeList])
            return unprobedNodeList[idx], False
        else:
            ## with probability 1-p, pick a node at random
            return np.random.choice(unprobedNodeList, 1)[0], True
    elif policy == 'globalmax_restart' or policy == 'globalmax_adaptive':
        prob = np.random.random()
        ## With probability p, follow global max
        if prob > p:
            idx = np.argmax(values[unprobedNodeList])
            return unprobedNodeList[idx], False
        else:
            ## with probability 1-p, pick a node at random
            ## Until there are none left, pick nodes from
            # the initial sample
            not_probed_initial = list(G.original_node_set - G.probedNodeSet)
            if len(not_probed_initial) > 0:
                node = np.random.choice(not_probed_initial, 1)[0]
                return G.node_to_row[node], True
            else:
                return np.random.choice(unprobedNodeList, 1)[0], True
    elif policy == 'globalmax_smartjump':
        prob = np.random.random()
        ## With probability p, follow global max
        if prob > p:
            idx = np.argmax(values[unprobedNodeList])
            return unprobedNodeList[idx], False
        else:
            ## with probability 1-p, pick a node at random
            ## proportional to the values
            probabilities = values[unprobedNodeList] + abs(min(values[unprobedNodeList])) + 1e-100
            probabilities /= sum(probabilities)
            #probabilities = utility.softmax(values[unprobedNodeList])
            return np.random.choice(unprobedNodeList, 1, p = probabilities)[0], True
    elif policy == 'globalrandom':
        idx = utility.random_pick(values[unprobedNodeList], 1)
        return  unprobedNodeList[idx[0]]
    else:
        logging.warning("Unrecognized policy \"" + policy + "\". Exiting.")
        sys.exit(1)

def RunIteration(G, alpha_input, episodes, epochs , initialNodes, Resultfile='output_file.txt', updateType = 'qlearning',
                 policy ='random', regularization = 'nonnegative', order = 'linear', reward_function = 'new_nodes', saveGAP = 0, current_iteration=0,
                 p = None, decay=0, k=4, target_attribute = None):
    theta_estimates = np.random.uniform(-0.2, 0.2,(G.get_numfeature(),))     # Initialize estimates at all 0.5
    initial_graph = G.copy()

    for episode in range(episodes):
        logging.info("episode: " + str(episode))
        probed_nodes, theta, rewards = RunEpisode(G, alpha_input, 0.0, 0.0, theta_estimates, \
                                                  epochs, Resultfile, updateType, policy, \
                                                  regularization, order, reward_function, saveGAP, episode, \
                                                  current_iteration, p, decay, k, target_attribute)

        theta_estimates = theta # Update value estimates
        G = initial_graph.copy() # reset G to the original sample
    # TODO this only returns the data for the final episode - is that what I want?
    return probed_nodes, theta, rewards