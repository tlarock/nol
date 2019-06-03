import sys
import numpy as np
import networkx as nx


def netdisc_sample(adjacency_list, attribute_dict, target_att, k):
    nodes = set()
    edges = set()
    sample_adjlist = dict()

    ## get the labeled nodes
    anomalous_nodes = [node for node, attr_lst in attribute_dict.items() if target_att in attr_lst]

    ## sample from the anomalous nodes
    sampled_anomalous = np.random.choice(anomalous_nodes, k, replace=False)
    ## add anomalous nodes and their neighbors to sample
    for node in sampled_anomalous:
        sample_adjlist.setdefault(node, dict())
        nodes.add(node)
        for ne in adjacency_list[node].keys():
            sample_adjlist.setdefault(ne, dict())
            sample_adjlist[node][ne] = dict()
            sample_adjlist[ne][node] = dict()
            nodes.add(ne)
            if (ne, node) not in edges:
                edges.add((node,ne))

    return sample_adjlist, nodes, edges


########################################################################
def generateNodeSample(complete_net, labels, p, label_num, interval):
    sample_network = nx.Graph()
    # desired_num_nodes = int(p * nx.number_of_nodes(network))
    desired_num_edges = int(p * nx.number_of_edges(complete_net))
    i = 1
    sample_num_edges = 0
    while sample_num_edges < desired_num_edges:
        print(interval*i)
        newnode = np.random.choice(complete_net.nodes(), interval * i)
        known_nodes = np.random.choice(labels.nonzero()[0], label_num, replace=False)
        newnode = np.append(newnode,[str(x) for x in known_nodes if str(x) in complete_net.nodes()])
        sample_network = nx.Graph(complete_net.subgraph(newnode))
        for known_node in known_nodes:
            node = str(known_node)
            sample_network.add_edges_from([(node,ne) for ne in complete_net[node].keys()])

        sample_num_edges = sample_network.number_of_edges()
        print("Picking ", interval * i, " random nodes with replacement")
        print("Current number of edges in sample: ", sample_num_edges)
        i = i + 1

        ## make sure the sample isn't too big! (more than 20%, arbitrarily)
        #if sample_num_edges > (desired_num_edges + desired_num_edges * 0.2):
        #    sample_num_edges = 0
        #    if np.random.random() > 0.5:
        #        i = i - 1
        #    else:
        #        i = i - 2
    print('KNOWN STARTING NODES')
    print(known_nodes)

    #nx.write_edgelist(sample_network, output_file, data=False)
    return sample_network, known_nodes

if __name__ == '__main__':
    args = sys.argv

    if True:
        input_dir = args[1]
        label_file = args[2]
        label_num = int(args[3])
        p = float(args[4])
        num_samples = int(args[5])
        interval = int(args[6])
    else:
        input_dir = '/Users/larock/git/rlnet/data/livejournal/network1/'
        label_file = '/Users/larock/git/rlnet/data/livejournal/truthLJ.txt'
        label_num = 5
        p = 0.0001
        num_samples = 1
        interval = 10000

    # Read in the node labels
    labels = []
    with open(label_file, 'r') as l:
        for line in l:
            labels.append(int(line))


    net = nx.read_edgelist(input_dir + 'network1.txt')

    labels = np.array(labels)
    for node in labels.nonzero()[0]:
        if str(node) not in net.nodes():
            labels[node] = 0

    for i in range(num_samples):
        sample, known_nodes= generateNodeSample(net, labels, p=p, label_num=label_num, interval=interval)

        nx.write_edgelist(sample, input_dir + 'network1_' + str(i) + '_node-sample_' + str(p) + '.txt', data=False)
