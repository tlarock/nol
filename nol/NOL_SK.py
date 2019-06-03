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
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from puAdapter import *


def RunEpisode(G, alpha, lambda_, gamma, theta, epochs, Resultfile='output_file.txt',
             updateType='qlearning', policy='logit', regularization='nonnegative', featureOrder='linear',
             reward_function='new_nodes', saveGap=0, episode=0, iteration=0, p = None, decay=0, target_attribute = None, burn_in=0):
    if policy not in ['high', 'low', 'rand']:
        features = G.calculate_features(G, featureOrder)
    ## TODO Adhoc
    values_list = []
    rewards_list = []
    probedNodes = []
    unprobedNodeSet = G.sample_node_set.copy()
    unprobedNodeIndices = {G.node_to_row[i] for i in unprobedNodeSet}
    graphSaveInterval = saveGap

    ## initialize log files
    intermediate_result_dir = os.path.join(Resultfile, 'intermediate_results')
    if not os.path.exists(intermediate_result_dir):
        os.makedirs(intermediate_result_dir)
    intermediate_name = os.path.join(intermediate_result_dir, policy + '_iter' + str(iteration) + '_episode_' + str(episode) + '_intermediate.txt')
    with open(intermediate_name, 'w+') as intermediateFile:
        intermediateFile.write('epoch\treward\testimate\tdelta\tjump\tp')
        if policy not in ['high', 'low', 'rand']:
            for i in range(theta.shape[0]):
                intermediateFile.write('\tTheta[' + str(i) + ']')
        intermediateFile.write('\n')

    intermediate_graph_dir = os.path.join(Resultfile, 'intermediate_graphs')
    if not os.path.exists(intermediate_graph_dir):
        os.makedirs(intermediate_graph_dir)
    intermediateGraphFile = os.path.join(intermediate_graph_dir, policy + '_iter' + str(iteration) + '_graph.txt')
    open(intermediateGraphFile, 'w').close()
    if policy not in ['high', 'low', 'rand']:
        featureFileDir = os.path.join(Resultfile, 'feature_analysis')
        if not os.path.exists(featureFileDir):
            os.makedirs(featureFileDir)

    ## initialize empty rewards list
    rewards = []

    ## If the reward function is on an attribute, some nodes are already queried
    targetNodeSet = set()
    if reward_function == 'attribute':
        for node in G.node_to_row:
            if target_attribute in G.attribute_dict[node]:
                if len(G.complete_graph_adjlist[node]) == len(G.sample_graph_adjlist[node]):
                    targetNodeSet.add(node)
                    rewards.append(1)
                    unprobedNodeSet.remove(node)
                    unprobedNodeIndices.remove(G.node_to_row[node])
                    G.probedNodeSet.add(node)
                    write_query(G, node, targetNodeSet, intermediateGraphFile)

        i = 0
        if policy not in ['high', 'low', 'rand']:
            for node in targetNodeSet:
                if i == 0:
                    samples_mat = np.append(features[G.node_to_row[node]], np.array([1]))
                else:
                    samples_mat = np.vstack( (samples_mat, np.append(features[G.node_to_row[node]], np.array([1])) ) )
                i+=1

        if policy == 'svm':
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


        initialTargetNodes = len(targetNodeSet)
        logging.info('# initial target nodes: ' + str(initialTargetNodes))
        numberOfTargetNodes = len(targetNodeSet) - initialTargetNodes

    count = 0
    interval = 500
    epoch = 0

    if burn_in <= 0:
        if policy == 'svm':
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

    ## burn in phase
    elif burn_in > 0:
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

            ## TODO Update sampled matrix
            if policy not in ['high', 'low', 'rand']:
                samples_mat = np.vstack( (samples_mat, np.append(features[nodeIndex], np.array([absoluteReward])) ) )
                features = G.update_features(G, probedNode, order=featureOrder)

            if policy == 'svm':
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


            write_intermediate(epoch, absoluteReward, 0, 0, 0, p, theta, intermediate_name)
            write_query(G, probedNode, targetNodeSet, intermediateGraphFile)
            if (saveGap != 0 and graphSaveInterval == (saveGap)) or epoch == (epochs-1):
                if epoch == (epochs - 1):
                    epoch += 1
                graphSaveInterval = 0
                if policy not in ['high', 'low', 'rand']:
                    featureFileName = policy + '_iter' + str(iteration) + '_features_' + str(epoch) + '.txt'
                    featureFile = os.path.abspath(os.path.join(featureFileDir, featureFileName))
                    write_features(G, features, featureFile)

            graphSaveInterval += 1
            epoch += 1

    ## run the actual experiment
    while epoch < epochs:
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

        ## Choose a node to probe
        nodeIndex, jump = action(G, policy, values, unprobedNodeIndices, p)
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

        ## Need the features of the current sample
        if policy not in ['high', 'low', 'rand']:
            ## TODO Update sampled matrix
            samples_mat = np.vstack( (samples_mat, np.append(features[nodeIndex], np.array([absoluteReward])) ) )
            features = G.F

        ## what is the value function of current state with current theta
        currentValue = values[nodeIndex]

        ## TODO compute a delta
        delta = currentValue - reward
        ## update the features    
        if policy not in ['high', 'low', 'rand']:
            features = G.update_features(G, probedNode, order=featureOrder)

        if policy == 'svm':
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



        ## TODO ad-hoc p decay
        if decay == 1:
            if epoch == 0:
                original_p = p
            p = original_p * (np.exp(-1*epoch/epochs))

        # writting intermediate numbers
        ## print a 1 if a random node was chosen, a 0 if the model was followed
        if jump is True:
            jval = 1
        elif jump is False:
            jval = 0
        else:
            logging.info('SOMETHING IS WRONG WITH JUMP SWITCH!')

        if policy not in ['high', 'low', 'rand']:
            write_intermediate(epoch, reward, currentValue, delta, jump, p, theta, intermediate_name)
        else:
            write_intermediate(epoch, reward, currentValue, delta, jump, p, None, intermediate_name)
        write_query(G, probedNode, targetNodeSet, intermediateGraphFile)
        if (saveGap != 0 and graphSaveInterval == (saveGap)) or epoch == (epochs-1):
            graphSaveInterval = 0
            if policy not in ['high', 'low', 'rand']:
                featureFileName = policy + '_iter' + str(iteration) + '_features_' + str(epoch) + '.txt'
                featureFile = os.path.abspath(os.path.join(featureFileDir, featureFileName))
                write_features(G, features, featureFile)

        graphSaveInterval += 1

        ## increment epoch
        epoch +=1

    intermediateFile.close()
    print(sum(rewards))

    logging.info('Total reward: ' + str(sum(rewards)))
    return probedNodes, theta, rewards


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


def write_intermediate(epoch, reward, currentValue, delta, jump, p, theta, intermediate_name):
    with open(intermediate_name, 'w+') as intermediateFile:
        # write intermediate numbers
        intermediateFile.write(str(epoch) + '\t' + str(reward) +  '\t' + str(currentValue)  + '\t'+ str(delta) + '\t' + str(jump) + '\t' + str(p))

        if theta is not None:
            for i in range(theta.shape[0]):
                intermediateFile.write('\t' + str(theta[i]))
        intermediateFile.write('\n')

def compute_svm_values(samples_mat, unprobed_features, unprobedNodeIndices):
    features = samples_mat[:,0:samples_mat.shape[1]-1]
    y = samples_mat[:,samples_mat.shape[1]-1]
    model = SVC(probability=False, kernel = 'poly' )
    model = model.fit(features, y)
    new_values =dict()
    for node in unprobedNodeIndices:
        new_values[node] = model.decision_function(unprobed_features[node].reshape(1,-1))
        #new_values[node] = model.predict_proba(unprobed_features[node].reshape(1,-1))[0][1]


    return new_values

def compute_knn_values(samples_mat, unprobed_features, unprobedNodeIndices):
    features = samples_mat[:,0:samples_mat.shape[1]-1]
    y = samples_mat[:,samples_mat.shape[1]-1]
    model = KNeighborsClassifier()
    model = model.fit(features, y)
    new_values =dict()
    for node in unprobedNodeIndices:
        #new_values[node] = model.decision_function(unprobed_features[node].reshape(1,-1))
        new_values[node] = model.predict_proba(unprobed_features[node].reshape(1,-1))[0][1]


    return new_values

def compute_logit_values(samples_mat, unprobed_features, unprobedNodeIndices, one_class=False):
    ## TODO if only_positive, need to pass in the unlabeled data
    features = samples_mat[:,0:samples_mat.shape[1]-1]
    y = samples_mat[:,samples_mat.shape[1]-1]
    model = LogisticRegression()

    if one_class:
        model = PUAdapter(model, hold_out_ratio=0.1)
        model.fit(features, y)
    else:
        model = model.fit(features, y)


    values_arr = model.predict_proba(unprobed_features)

    if one_class:
        values = {node:values_arr[node] for node in unprobedNodeIndices}
        new_theta = np.zeros(features.shape[1])
    else:
        values = {node:values_arr[node][1] for node in unprobedNodeIndices}
        new_theta = np.array(model.coef_).T


    new_theta = new_theta.reshape((new_theta.shape[0],))
    return values, new_theta

def compute_linreg_values(samples_mat, unprobed_features, unprobedNodeIndices):
    features = samples_mat[:,0:samples_mat.shape[1]-1]
    y = samples_mat[:,samples_mat.shape[1]-1]
    model = LinearRegression()
    model = model.fit(features, y)

    new_theta = np.array(model.coef_).T

    values_arr = unprobed_features.dot(new_theta)

    values = {node:values_arr[node] for node in unprobedNodeIndices}
    return values, new_theta

def compute_deg_values(G, unprobedNodeIndices):
    deg_values = {node:len(G.sample_graph_adjlist[G.row_to_node[node]]) for node in unprobedNodeIndices}
    return deg_values

def action(G, policy, values, unprobedNodeIndices, p = -1):
    """
    Global random jump policy, meaning any node can be probed
    at any time, and with probabiliy 1-p a uniformly random
    node is selected.
    """
    idx = []
    restart_probability = utility.getProbRestart()
    unprobedNodeList = [row for row in unprobedNodeIndices]
    prob = np.random.random()
    if prob > p and policy != 'rand':
        if policy == 'low':
            idx = min(values.items(), key=lambda kv: kv[1])[0]
        #elif policy == 'high':
        else:
            ## With probability p, follow global max
            idx = max(values.items(), key=lambda kv: kv[1])[0]
        return idx, False
    else:
        ## with probability 1-p, pick a node at random
        idx = np.random.choice(unprobedNodeList, 1)[0]
        return idx, True

def RunIteration(G, alpha_input, episodes, epochs , initialNodes, Resultfile='output_file.txt', updateType = 'qlearning',
                 policy ='rand', regularization = 'nonnegative', order = 'linear', reward_function = 'new_nodes', saveGAP = 0, current_iteration=0,
                 p = None, decay=0, target_attribute = None, burn_in=0):
    if policy not in ['high', 'low', 'rand']:
        theta_estimates = np.random.uniform(-0.2, 0.2,(G.get_numfeature(),))     # Initialize estimates at all 0.5
    else:
        theta_estimates = 0

    initial_graph = G.copy()
    print(policy)
    for episode in range(episodes):
        logging.info("episode: " + str(episode))
        probed_nodes, theta, rewards = RunEpisode(G, alpha_input, 0.0, 0.0, theta_estimates, \
                                                  epochs, Resultfile, updateType, policy, \
                                                  regularization, order, reward_function, saveGAP, episode, \
                                                  current_iteration, p, decay, target_attribute,burn_in)

        theta_estimates = theta # Update value estimates
        G = initial_graph.copy() # reset G to the original sample
    # TODO this only returns the data for the final episode - is that what I want?
    return probed_nodes, theta, rewards
