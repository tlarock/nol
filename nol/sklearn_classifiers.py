from scipy import stats
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from puAdapter import *

def compute_svm_values(samples_mat, unprobed_features, unprobedNodeIndices, one_class=False):
    features = samples_mat[:,0:samples_mat.shape[1]-1]
    y = samples_mat[:,samples_mat.shape[1]-1]

    if one_class:
        model = SVC(probability=True, kernel = 'poly')
        model = PUAdapter(model, hold_out_ratio=0.1)
        model.fit(features, y)
    else:
        model = SVC(probability=False, kernel = 'poly')
        model = model.fit(features, y)

    new_values =dict()
    for node in unprobedNodeIndices:
        if one_class:
            new_values[node] = model.predict_proba(unprobed_features[node].reshape(1,-1))
        else:
            new_values[node] = model.decision_function(unprobed_features[node].reshape(1,-1))

    return new_values

def compute_knn_values(samples_mat, unprobed_features, unprobedNodeIndices, one_class=False):
    features = samples_mat[:,0:samples_mat.shape[1]-1]
    y = samples_mat[:,samples_mat.shape[1]-1]
    model = KNeighborsClassifier()

    if one_class:
        model = PUAdapter(model, hold_out_ratio=0.1)
        model.fit(features, y)
    else:
        model = model.fit(features, y)

    new_values =dict()
    for node in unprobedNodeIndices:
        new_values[node] = model.predict_proba(unprobed_features[node].reshape(1,-1))[0][1]

    return new_values

def compute_logit_values(samples_mat, unprobed_features, unprobedNodeIndices, one_class=False):
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

def compute_linreg_values(samples_mat, unprobed_features, unprobedNodeIndices, one_class=False):
    features = samples_mat[:,0:samples_mat.shape[1]-1]
    y = samples_mat[:,samples_mat.shape[1]-1]
    model = LinearRegression()

    if one_class:
        model = PUAdapter(model, hold_out_ratio=0.1)
        model.fit(features, y)
    else:
        model = model.fit(features, y)

    new_theta = np.array(model.coef_).T

    values_arr = unprobed_features.dot(new_theta)

    values = {node:values_arr[node] for node in unprobedNodeIndices}
    return values, new_theta

def compute_deg_values(G, unprobedNodeIndices):
    deg_values = {node:len(G.sample_graph_adjlist[G.row_to_node[node]]) for node in unprobedNodeIndices}
    return deg_values

