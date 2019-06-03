#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import Network, NOL, NOL_HTR, NOL_SK
from generate_netdisc_samples import netdisc_sample
import numpy as np
import os
import re
import glob
import argparse
import logging
from multiprocessing import Pool
from utility import read_network
from utility import read_attributes

MODELS=['globalrandom_jump','localrandom_jump','globalmax','globalmax_jump', 'globalmax_restart',
        'localmax','globalrandom','localrandom','globalmax_smartjump', 'globalmax_adaptive', 'NOL-HTR', 'logit', 'svm', 'knn', 'linreg', 'high', 'low', 'rand']
FEATURES=['netdisc', 'default', 'refex', 'node2vec', 'n2v-refex', 'knn']



def runOneTrial(model, sample_dir, realAdjList, sampleType, samplePortion, alpha, lam,
                gamma, episodes, epochs, outfile, ite, saveGAP, feature_type, reward_function, p, decay, k, attribute_dict,
                target_attribute, burn_in, compute_sample):

    logger = logging.getLogger(__name__)
    logger.info(str(ite))
    if not compute_sample:
        sample = os.path.join(sample_dir,'*_'+str(ite)+'_'+str(sampleType)+'-*'+str(samplePortion)+'*')
        sampleAdjList, nodes, edges = read_network(min(glob.glob(sample), key=len))
    else:
        sampleAdjList, nodes, edges = netdisc_sample(realAdjList, attribute_dict, target_attribute, 5)

    logger.info("Starting sample nodes: " + str(len(nodes)))
    logger.info("Starting sample edges: " + str(len(edges)))

    if model not in ['high', 'low', 'rand']:
        g = Network.Network(realAdjList, sampleAdjList, calculate_features=True, feature_type = feature_type, attribute_dict=attribute_dict)
    else:
        g = Network.Network(realAdjList, sampleAdjList, calculate_features=False, attribute_dict=attribute_dict)


    if model == 'NOL-HTR':
        probednode, _, rewards = NOL_HTR.RunIteration(g, alpha, episodes, epochs, list(nodes), outfile, 'sarsa', model, 'no',\
                                                            reward_function = reward_function, saveGAP = saveGAP, current_iteration=ite, p=p, k=k,\
                                                            decay=decay, target_attribute=target_attribute)
    elif model == 'svm' or model == 'knn' or model == 'linreg' or model == 'logit' or model == 'high' or model == 'low' or model == 'rand':
        probednode, _, rewards = NOL_SK.RunIteration(g, alpha, episodes, epochs, list(nodes), outfile, 'sarsa', model, 'no', reward_function = reward_function, saveGAP = saveGAP, current_iteration=ite, p=p, decay=decay, target_attribute=target_attribute, burn_in=burn_in)
    else:
        probednode, _, rewards = NOL.RunIteration(g, alpha, lam, gamma, episodes, epochs, list(nodes), outfile, 'sarsa', model, 'no', reward_function = reward_function, saveGAP = saveGAP, current_iteration=ite, p=p, target_attribute=target_attribute)


    reward_cumulative = np.cumsum(rewards)
    reward_sd = []
    for i in range(1, len(reward_cumulative) + 1):
        reward_sd.append(np.std(reward_cumulative[:i]))

    return reward_cumulative


def runManyTrials(model, input_file, sample_type, sample_size, sample_dir, output_dir, budget, episodes, iterations, save_gap, alpha, lam, gamma,
                  feature_type, reward_function, p, decay, k, attribute_file, target_attribute, burn_in, compute_sample, processes):

    logger = logging.getLogger(__name__)

 
    ## The complete network
    realAdjList, nodes, edges = read_network(input_file)

    attribute_dict = None
    if attribute_file:
        attribute_dict = read_attributes(attribute_file)
        if target_attribute:
            logger.info('Attributes specified without target.')
        else:
            target_size = 0
            for u, l in attribute_dict.items():
                if target_attribute in l:
                    target_size += 1
            logger.info(str(target_size) + ' nodes with attribute targets.')

    ## matrix of results per iteration
    results_matrix = []

    if sample_type != 'compute':
        sample_dir_files = [f for f in os.listdir(sample_dir) if str('_' + str(sample_type) + '-sample_' + str(sample_size)) in f]
    else:
        ## TODO adhoc way to choose to compute samples
        sample_dir_files = ['compute']*iterations

    ## filename for averaged output for this realization
    final_table = os.path.join(output_dir, model + '_a' + str(alpha) + '_l' + str(lam) + '_g' + str(gamma) + '.csv')

    logger.info("Running Model: " + model)
    logger.info("Starting network nodes: " + str(len(nodes)))
    logger.info("Starting network edges: " + str(len(edges)))
    row = 0
    if processes == 1:
        for i in range(iterations):
            logger.info("Iteration: " + str(i))
            result = runOneTrial(model,sample_dir,realAdjList,sample_type,sample_size, alpha,lam,gamma,episodes,budget, output_dir,i,save_gap,\
                                 feature_type, reward_function, p, decay, k, attribute_dict, target_attribute, burn_in, compute_sample)

            results_matrix.append(np.array(result))
            # Compute and output averages after every iteration (so as to not lose data if
            # experiment needs to be stopped prematurely)
            if row > 0:
                tmp_result = np.array(results_matrix)
                results_avg = np.mean(tmp_result, axis=0)
                results_sd = np.std(tmp_result, axis=0)
            else:
                results_avg = results_matrix[0]
                results_sd = np.zeros(len(results_matrix[0]))
            with open(final_table, 'w') as final_result:
                N = float(len(realAdjList.keys()))
                final_result.write('Probe\tAvgRewards\tStdRewards\tProbeFrac\n')
                for b in range(budget):
                    final_result.write(str(b) + '\t' + str(results_avg[b]) + '\t' + str(results_sd[b]) + '\t' + str((b+1) / N) + '\n')


            row += 1
    else:
        pool = Pool(processes)
        arguments = [(model,sample_dir,realAdjList,sample_type,sample_size, alpha,lam,gamma,episodes,budget, output_dir,i,save_gap,\
                                 feature_type, reward_function, p, decay, k, attribute_dict, target_attribute, burn_in, compute_sample) for i in range(iterations)]
        results_matrix = pool.starmap(runOneTrial, arguments)
        results_avg = np.mean(results_matrix, axis=0)
        results_sd = np.std(results_matrix, axis=0)
        with open(final_table, 'w') as final_result:
            N = float(len(realAdjList.keys()))
            final_result.write('Probe\tAvgRewards\tStdRewards\tProbeFrac\n')
            for b in range(budget):
                final_result.write(str(b) + '\t' + str(results_avg[b]) + '\t' + str(results_sd[b]) + '\t' + str((b+1) / N) + '\n')


    return results_matrix

def experiment(model, input_directory, sample_folder, output_folder, \
               budget, episodes, iterations, networks, save_gap, \
               alpha, lambda_, gamma, feature_type, reward_function, p, decay, k, attribute_file, target_attribute, burn_in, compute_sample, processes):
    ## Get sample information
    type_regex = re.compile(r'[a-z]*-')
    size_regex = re.compile(r'[0-9][.][0-9]+')

    sample_type = re.findall(type_regex, sample_folder)[0].split('-')[0]
    sample_size = re.findall(size_regex, sample_folder)[0]

    if compute_sample == 'True':
        compute_sample = True
    elif compute_sample == 'False':
        compute_sample = False

    ## For every network
    for i in range(1, networks+1):
        logging.info('network: ' + str(i))
        ## Generate the input, sample file and output directory names
        input_file = input_directory + 'network' + str(i) + '/network' + str(i) + '.txt'
        sample_dir = input_directory + 'network' + str(i) + '/'
        output_dir = output_folder + 'network' + str(i)

        ## Run 'iterations' iterations on this network
        results_matrix = runManyTrials(model, input_file, sample_type, sample_size, sample_dir, output_dir, \
               budget, episodes, iterations, save_gap, \
               alpha, lambda_, gamma, feature_type, reward_function, p, decay, k, attribute_file, target_attribute, burn_in, compute_sample, processes)


def main(args):
    log_file = "/home/larock.t/git/nol/results/logs/outLTD_" + str(args.model) + "_" + str(args.gamma) + "_" + \
            str(args.lambda_) + "_" + str(args.iterations) + "_" + str(args.budget)+ ".out"
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    if args.ktype == 'int':
        k = int(args.k)
    elif args.ktype == 'funct':
        k = eval(args.k)
    experiment(args.model, args.input_directory, args.sample_folder, args.output_folder,
                  args.budget, args.episodes, args.iterations, args.networks, args.save_gap,
                  args.alpha, args.lambda_, args.gamma, args.feature_type, args.reward_function, args.p, args.decay, k, args.attribute_file,
               args.target_attribute, args.burn_in, args.compute_sample, args.processes)
    logging.info("END")

if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser(description="Run Linear TDL Experiments")
    parser.add_argument('-m', dest='model', choices=MODELS, help='type of RL model to use')
    parser.add_argument('--attr', dest='attribute_file', default=None, help='Location of attribute file')
    parser.add_argument('--target', dest='target_attribute', default = None, type=int, help='Target attribute (row in attribute file')
    parser.add_argument('-i', dest='input_directory', help='directory containing complete graph\'s adjacency list. Graphs should have name \'networkA\', where A indicates the realization #')
    parser.add_argument('-s', dest='sample_folder', help='name of directory that holds the sample files')
    parser.add_argument('-o', dest='output_folder', help='name of the desired output directory')
    parser.add_argument('-n', dest='networks', type = int, help = 'number of networks to run experiments on')
    parser.add_argument('-iter', dest='iterations', type = int, help='number of iterations')
    parser.add_argument('-e', dest='episodes', type=int, default=1, help='number of episodes')
    parser.add_argument('-b', dest='budget', type=int, help='budget of probes/number of epochs')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0, help='learning rate')
    # Underscore in lambda_ to prevent syntax conflict with Python lambda
    parser.add_argument('--lamb', dest='lambda_', type=float, default=0, metavar='lambda', help='trace parameter')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0, help='discount factor')
    parser.add_argument('--feats', dest='feature_type', choices=FEATURES,
                        help='default, refex or node2vec features')
    parser.add_argument('--reward', dest='reward_function', choices=['new_nodes', 'new_edges', 'nodes_and_triangles', 'new_nodes_local', 'attribute'],
                        help='new_nodes, new_edges or nodes_and_triangles reward function.')
    parser.add_argument('--save_gap', dest='save_gap', type=int, default=0, help='gap (in epochs) between intermediate file saves')
    parser.add_argument('-p', dest='p', type=float, default=0.3, help='Probability of random jump in jump strategy.')
    parser.add_argument('--decay', dest='decay', type=int, default=0, help='Exponential decay on epsilon?')
    parser.add_argument('--ktype', dest='ktype', default='int', choices=['funct', 'int', 'delta'])
    parser.add_argument('-k', dest='k', type=str, default=1, help='k for NOL-HTR.')
    parser.add_argument('--burn', dest='burn_in', type=int, default=0, help='# of high degree burn-in pulls to make')
    parser.add_argument('--sample', dest='compute_sample', type=str, default='False', choices=['True', 'False'], help='Flag to compute the sample rather than read it.')
    parser.add_argument('--processes', dest='processes', type=int, default=1, help='# of proceses to use (default 1, no multiproc)')

    main(parser.parse_args())
