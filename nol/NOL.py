#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import Network
import sys
import os
import logging


def RunEpisode(G, alpha, theta, epochs, Resultfile='output_file.txt', policy='NOL', regularization='nonnegative', featureOrder='linear',
             reward_function='new_nodes', saveGap=0, episode=0, iteration=0, p = None, target_attribute = None):
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
        targetNodeSet = {node for node in G.node_to_row \
                         if target_attribute in G.attribute_dict[node] and (len(G.complete_graph_adjlist[node]) == len(G.sample_graph_adjlist[node]))}
        initialTargetNodes = len(targetNodeSet)
        print('initial target nodes: ' + str(initialTargetNodes))
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

    count = 0
    interval = 500
    for epoch in range(epochs):
        ## TODO
        values_list.append(values[list(unprobedNodeIndices)])
        rewards_list.append([(len(G.complete_graph_adjlist[node]) - len(G.sample_adjlist_sets[node])) for node in unprobedNodeSet])


        if count == interval:
            count = 0
            logging.info('Iteration: ' + str(iteration) + ' Epoch: ' + str(epoch))
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

        if epoch == 0 and reward_function == 'attribute':
            rewards[0] += initialTargetNodes

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
        if len(unprobedNodeIndices) == 0:
            break

        ## Update (q-learning)
        next_node, _ = action(G, policy, values, unprobedNodeIndices, p)
        nextValue = values[next_node]

        ## Calculate the estimated future reward
        #estimatedReward = reward + gamma * nextValue
        estimatedReward = reward + nextValue
        delta = estimatedReward - currentValue

        ## Update the eligibility trace
        #eligibilityTraces = lambda_ * gamma * eligibilityTraces + currentGradient.T

        ## Do learning
        old_theta = theta
        if alpha > 0:
            ## regular learning with positive learning rate
            #theta = theta + (alpha * delta * eligibilityTraces)
            theta = theta + alpha*delta*currentGradient.T
        else:
            logging.warning("Learning rate Alpha can not be 0")
            sys.exit(1)


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
            intermediateFile.write(str(epoch) + '\t' + str(estimatedReward) + '\t' + str(currentValue)  + '\t'+ str(regret) + '\t' +  str(delta) + '\t' + str(reward) + '\t' + str(jump) + '\t' + str(MSE) + '\t' + str(theta_diff))

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
    elif policy == 'NOL':
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


def RunIteration(G, alpha_input, episodes, epochs , initialNodes, Resultfile='output_file.txt', policy='NOL', regularization = 'nonnegative', order = 'linear', reward_function = 'new_nodes', saveGAP = 0, current_iteration=0, p = None, target_attribute = None):
    theta_estimates = np.random.uniform(-0.2, 0.2,(G.get_numfeature(),))     # Initialize estimates at all 0.5
    initial_graph = G.copy()

    for episode in range(episodes):
        logging.info("episode: " + str(episode))
        probed_nodes, theta, rewards = RunEpisode(G, alpha_input, theta_estimates, epochs, Resultfile, policy, regularization, order, reward_function, saveGAP, episode, current_iteration, p, target_attribute)

        theta_estimates = theta # Update value estimates
        G = initial_graph.copy() # reset G to the original sample
    return probed_nodes, theta, rewards
