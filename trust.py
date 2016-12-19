from __future__ import division

import re
import sys
import igraph
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from transform_funcs import *
from scipy.sparse import coo_matrix

def graph_to_sparse_matrix(G):
    n = G.vcount()
    xs, ys = map(np.array, zip(*G.get_edgelist()))
    if not G.is_directed():
        xs, ys = np.hstack((xs, ys)).T, np.hstack((ys, xs)).T
    else:
        xs, ys = xs.T, ys.T
    return coo_matrix((np.ones(xs.shape), (xs, ys)), shape=(n, n), dtype=np.int16)

def get_feature(G, f):
    return _transform_func_degree(getattr(G, f)()) if callable(getattr(G, f)) else getattr(G, f)

# aggregate by the mean value of feature of neighbours
def mean_neighbour(A, d, feature):
    return A.dot(feature) / d

def get_feature_matrix(G, features, rounds=5):
    # local clustering coefficient
    G_sim = G.as_directed().simplify(multiple=False)                 # remove loops

    lcc = np.array(G_sim.transitivity_local_undirected(mode='zero'))
    lcc[lcc < 0] = 0                                                 # workaround of iGraph bug
    G.clustering_coefficient = lcc

    # compute PageRank
    G_sim = G.copy()
    G_sim = G_sim.simplify(multiple=False)                           # remove loops

    alpha = 0.15
    pagerank = np.array(G_sim.pagerank(damping=1-alpha))
    n = G_sim.vcount()
    pr_ub = np.percentile(pagerank, 100-300/n, interpolation='lower')
    pagerank[pagerank > pr_ub] = pr_ub
    min_max_scaler = MinMaxScaler()
    pagerank = min_max_scaler.fit_transform(pagerank)
    #dangling_nodes = (np.array(G_sim.outdegree()) == 0)
    #r_low = (alpha + (1-alpha) * sum(pagerank[dangling_nodes])) / G_sim.vcount()
    #pagerank = pagerank / r_low
    G.pr = pagerank

    feature_matrix = [ get_feature(G, f) for f in features ]
    X = np.array(feature_matrix).T

    # adjacency matrix (simplified)
    A = graph_to_sparse_matrix(G.as_undirected().simplify())
    d = np.squeeze(np.array(A.sum(axis=1))).astype(np.int)
    d[d == 0] = 1

    for i in range(rounds):
        feature_matrix = [ mean_neighbour(A, d, f) for f in feature_matrix ]
        X = np.concatenate((X, np.array(feature_matrix).T), axis=1)

    #X = np.hstack((X, np.array([pagerank]).T))
    return X

def read_data(features, f_network, f_roles=''):
    # dataset (network)
    df = pd.read_csv(f_network, header=None)

    nodes = np.unique(df[[0, 1]].values);
    max_node_num = max(nodes) + 1
    num_nodes = len(nodes)

    G = igraph.Graph(directed=True)
    G.add_vertices(max_node_num)
    G.add_edges(df[[0, 1]].values)

    G = G.subgraph(nodes)

    # features
    X = get_feature_matrix(G, features)

    # dataset (roles)
    r = [0] * max_node_num

    if f_roles != '':
        df_role = pd.read_csv(f_roles, header=None)
        roles = df_role[[0,1]].values

        for role in roles:
            if role[0] < max_node_num:
                r[role[0]] = role[1]

        r = [r[i] for i in nodes]

    return np.squeeze(X), np.array(r)

# main
def main():
    # positive roles
    roles = [ 'Administrator', 'Moderator', '^Moderator', 'Subscriber' ]
    features = [ 'clustering_coefficient' , 'degree' , 'indegree' , 'outdegree' , 'pr' ]

    # read datasets
    X_source, r_source = read_data(features, f_net_source, f_role_source)
    X_target, _ = read_data(features, f_net_target)
    trust_target = np.zeros(len(X_target))

    df_dict = pd.read_csv(f_dictionary, sep=' ')

    for r in roles:
        # one-class classification for role r
        r_regex = re.compile(r)
        y_source = [ bool(re.search(r_regex, role)) for role in r_source ]

        # classifier
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_source, y_source)

        y_target_proba = clf.predict_proba(X_target)[:,1]
        y_target_proba = y_target_proba / max(y_target_proba)
        trust_target = trust_target + y_target_proba

    # negative role : Banned
    r = 'Banned'
    # one-class classification for role r
    r_regex = re.compile(r)
    y_source = [ bool(re.search(r_regex, role)) for role in r_source ]

    # classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_source, y_source)

    y_target_proba = clf.predict_proba(X_target)[:,1]
    y_target_proba = y_target_proba / max(y_target_proba)
    trust_target = trust_target - y_target_proba
    trust_target /= len(roles)

    ids = df_dict['ent.string.name'].values[range(len(trust_target))]
    df_out = pd.DataFrame({'id':ids, 'trust': trust_target})
    df_out.to_csv(f_out, index=False, header=False)

if __name__ == '__main__':
    # init
    ## network files
    f_net_source = sys.argv[1]
    f_net_target = sys.argv[2]

    ## role file
    f_role_source = sys.argv[3]

    ## dictionary of target (UID to UID)
    f_dictionary = sys.argv[4]

    ## output file (trust in SAG dataset)
    f_out = sys.argv[5]

    _transform_func_degree = degree_transform

    main()

