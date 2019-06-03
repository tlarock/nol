import subprocess
import numpy as np
import os

def calculate_features(self, order='linear'):
    """
    Calculates node2vec embedding for features.
    """
    ## output the graph
    randnum = str(np.random.rand())
    outfile = 'tmp' + randnum + '.out'
    with open(outfile, 'w') as tmp_out:
        for node in self.sample_graph_adjlist:
            for neighbor in self.sample_adjlist_sets[node]:
                tmp_out.write(node + ' ' + neighbor + '\n')

    ## run the embedding
    embfile = 'tmp' + randnum + '.emb'
    subprocess.run(['../../snap/examples/node2vec/node2vec', '-i:' + str(outfile), '-o:' + str(embfile), '-l:3', '-d:24', '-p:0.3'])

    ## read feautures
    with open(embfile, 'r') as tmp_in:
        ## need to skip the first line
        first = True
        for line in tmp_in:
            if first:
                array = line.split(' ')
                NumF = int(array[1])
                first = False
                ## initialize the feature matrix
                features = np.ones( (len(self.sample_node_set), NumF))
            else:
                array = line.split(' ')
                row = self.node_to_row[array[0]]
                assert len(array) == (NumF + 1), 'len(array): ' + str(len(array))
                features[row] = np.array([float(array[i]) for i in range(1, len(array))])

    self.NumF = features.shape[1]
    self.F = features

    os.remove(outfile)
    os.remove(embfile)

    return features

def update_features(self, node, order='linear'):
    """
    Updates the feature matrix based on the node being probed.
    """
    return calculate_features(self, order)
