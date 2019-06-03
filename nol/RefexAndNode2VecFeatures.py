import numpy as np
import Node2VecFeatures as n2v
import RefexFeatures as refex

def calculate_features(self, order = 'linear'):
    refex_feats = refex.calculate_features(self, order)
    node2vec_feats = n2v.calculate_features(self, order)
    features = np.concatenate((refex_feats, node2vec_feats), axis = 1)
    self.NumF = features.shape[1]
    self.F = features
    return features


def update_features(self, node, order='linear'):
    refex_feats = refex.update_features(self, node, order)
    node2vec_feats = n2v.update_features(self, node, order)
    features = np.concatenate((refex_feats, node2vec_feats), axis = 1)
    self.NumF = features.shape[1]
    self.F = features
    return features
