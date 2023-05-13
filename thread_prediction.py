'''
Python file for testing different Machine Learning models' accuracy to predict whether a Reddit thread is discussion or non-discussion based using different metrics as the feature vectors.

Author: Mikko Lempinen
'''

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from karateclub.dataset import GraphSetReader
from karateclub import FeatherGraph
from karateclub import FGSD
from karateclub import GeoScattering
from karateclub import GL2Vec
from karateclub import Graph2Vec
from karateclub import IGE
from karateclub import LDP
from karateclub import NetLSD
from karateclub import SF
from karateclub import WaveletCharacteristic
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.random_projection import *
from sklearn.metrics import *
import lightgbm as lgb


def plot_degree_distributions(graphdata):
    '''
    Plot each graph in a given list of graphs and plot the degree distribution histogram for said graph
    :param graphdata: (list) List of NetworkX Graphs
    '''
    for i in graphdata:
        G = nx.Graph(i)
        degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
        fig = plt.figure("Degree of a graph", figsize=(8, 8))
        axgrid = fig.add_gridspec(5, 4)

        ax0 = fig.add_subplot(axgrid[0:3, :])
        Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
        pos = nx.spring_layout(Gcc, seed=10396953)
        nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
        nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
        ax0.set_title("Connected components of G")
        ax0.set_axis_off()

        ax2 = fig.add_subplot(axgrid[3:, :])
        ax2.bar(*np.unique(degree_sequence, return_counts=True))
        ax2.set_title("Degree histogram")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("# of Nodes")

        fig.tight_layout()
        plt.show()
        #deg_values, deg_counts = np.unique(degree_sequence, return_counts=True)


def deg_feature_vectors(graphdata):
    '''
    Create degree distribution feature vectors for a given graph data set.
    Feature vector is a fixed vector with the size of 10, where [Amount of nodes with degree of 1, 
    Amount of nodes with degree of 2, Amount of nodes with degree of 3, Amount of nodes with degree of 4, 
    Amount of nodes with degree of 5, Amount of nodes with degree of 6, Amount of nodes with degree of 7,
    Amount of nodes with degree of 8, Amount of nodes with degree of 9, Amount of nodes with degree of 10+].
    :param graphdata: (list) Data of each graph to generate a feature vector for.
    :return: (list) Feature vectors.
    '''
    feature_vectors = []
    for i in graphdata:
        feature_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        G = nx.Graph(i)
        degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
        deg_values, deg_counts = np.unique(degree_sequence, return_counts=True)
        for i, j in enumerate(deg_values):
            if j == 1:
                feature_vector[0] = deg_counts[i]
            if j == 2:
                feature_vector[1] = deg_counts[i]
            if j == 3:
                feature_vector[2] = deg_counts[i]
            if j == 4:
                feature_vector[3] = deg_counts[i]
            if j == 5:
                feature_vector[4] = deg_counts[i]
            if j == 6:
                feature_vector[5] = deg_counts[i]
            if j == 7:
                feature_vector[6] = deg_counts[i]
            if j == 8:
                feature_vector[7] = deg_counts[i]
            if j == 9:
                feature_vector[8] = deg_counts[i]
            if j == 10:
                feature_vector[9] = deg_counts[i]
            if j == 11:
                feature_vector[10] = deg_counts[i]
            if j == 12:
                feature_vector[11] = deg_counts[i]
            if j == 13:
                feature_vector[12] = deg_counts[i]
            if j == 14:
                feature_vector[13] = deg_counts[i]
            if j > 14:
                feature_vector[14] += deg_counts[i]
        #print(f"Feature vector: {feature_vector}")
        feature_vectors.append(feature_vector)
    return feature_vectors

def betweenness_centrality_vectors(graphdata):
    '''
    Create betweenness centrality feature vectors for a given graph data set.
    '''
    feature_vectors = []
    for i in graphdata:
        feature_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        G = nx.Graph(i)
        bc = nx.betweenness_centrality(G)
        for key in bc:
            if bc[key] == 0.0:
                feature_vector[0] += 1
            if 0.05 > bc[key] > 0.0:
                feature_vector[1] += 1
            if 0.1 > bc[key] >= 0.05:
                feature_vector[2] += 1
            if 0.2 > bc[key] >= 0.1:
                feature_vector[3] += 1
            if 0.3 > bc[key] >= 0.2:
                feature_vector[4] += 1
            if 0.4 > bc[key] >= 0.3:
                feature_vector[5] += 1
            if 0.5 > bc[key] >= 0.4:
                feature_vector[6] += 1
            if 0.6 > bc[key] >= 0.5:
                feature_vector[7] += 1
            if 0.7 > bc[key] >= 0.6:
                feature_vector[8] += 1
            if 0.8 > bc[key] >= 0.7:
                feature_vector[9] += 1
            if 0.9 > bc[key] >= 0.8:
                feature_vector[10] += 1
            if 1 > bc[key] >= 0.9:
                feature_vector[11] += 1
            if bc[key] == 1.0:
                feature_vector[12] += 1
        feature_vectors.append(feature_vector)
    return feature_vectors

def closeness_centrality_vectors(graphdata):
    '''
    Create closeness centrality feature vectors for a given graph data set.
    '''
    feature_vectors = []
    for i in graphdata:
        feature_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        G = nx.Graph(i)
        cc = nx.closeness_centrality(G)
        for key in cc:
            if cc[key] == 0.0:
                feature_vector[0] += 1
            if 0.05 > cc[key] > 0.0:
                feature_vector[1] += 1
            if 0.1 > cc[key] >= 0.05:
                feature_vector[2] += 1
            if 0.2 > cc[key] >= 0.1:
                feature_vector[3] += 1
            if 0.3 > cc[key] >= 0.2:
                feature_vector[4] += 1
            if 0.4 > cc[key] >= 0.3:
                feature_vector[5] += 1
            if 0.5 > cc[key] >= 0.4:
                feature_vector[6] += 1
            if 0.6 > cc[key] >= 0.5:
                feature_vector[7] += 1
            if 0.7 > cc[key] >= 0.6:
                feature_vector[8] += 1
            if 0.8 > cc[key] >= 0.7:
                feature_vector[9] += 1
            if 0.9 > cc[key] >= 0.8:
                feature_vector[10] += 1
            if 1 > cc[key] >= 0.9:
                feature_vector[11] += 1
            if cc[key] >= 1.0:
                feature_vector[12] += 1
        feature_vectors.append(feature_vector)
    return feature_vectors


# Initialize graph reader
reader = GraphSetReader("reddit10k")

# Reddit thread data
graphs = reader.get_graphs()
y = reader.get_target()
# 10 Train-test random state values for deterministic set selections
random_states = [0, 1, 5, 12, 38, 42, 47, 49, 72, 77]

# Generate feature vectors for each graph (Uncomment one of these to use)
#feature_vectors, index = deg_feature_vectors(graphs), 0
#feature_vectors, index = betweenness_centrality_vectors(graphs), 1
feature_vectors, index = closeness_centrality_vectors(graphs), 2

if index == 0:
    fv = "Degree Distribution"
if index == 1:
    fv = "Betweenness Centralities"
if index == 2:
    fv = "Closeness Centralities"
print(f"\n--------------------------AUC Scores using node {fv} as feature vectors------------------------------------\n")
# Predicting threads with Logistic Regression
auc_mean = 0
counter = 0
for i in random_states:
    # 80/20 Train-test with degree distribution as predictors of the thread type
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, y, test_size=0.2, random_state=i)
    # Use training data to predict the probability of a thread being discussison based with a logistic regression model
    downstream_model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    y_hat = downstream_model.predict_proba(X_test)[:, 1]
    # Evaluate model performance
    auc = roc_auc_score(y_test, y_hat)
    auc_mean += auc
    counter += 1
auc_mean = auc_mean / counter
print(f"Logistic Regression Average AUC: {auc_mean:.4f}")

# Predicting threads with Linear Dictriminant Analysis
auc_mean = 0
counter = 0
for i in random_states:
    # 80/20 Train-test with degree distribution as predictors of the thread type
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, y, test_size=0.2, random_state=i)
    # Use training data to predict the probability of a thread being discussison based with Linear Discriminant Analysis
    downstream_model = LinearDiscriminantAnalysis().fit(X_train, y_train)
    y_hat = downstream_model.predict_proba(X_test)[:, 1]
    # Evaluate model performance
    auc = roc_auc_score(y_test, y_hat)
    auc_mean += auc
    counter += 1
auc_mean = auc_mean / counter
print(f"Linear Dictriminant Analysis Average AUC: {auc_mean:.4f}")

# Predicting threads with K-Nearest Neighbors
auc_mean = 0
counter = 0
for i in random_states:
    # 80/20 Train-test with degree distribution as predictors of the thread type
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, y, test_size=0.2, random_state=i)
    # Use training data to predict the probability of a thread being discussison based with a K-Neighbors Classifier
    downstream_model = KNeighborsClassifier().fit(X_train, y_train)
    y_hat = downstream_model.predict_proba(X_test)[:, 1]
    # Evaluate model performance
    auc = roc_auc_score(y_test, y_hat)
    auc_mean += auc
    counter += 1
auc_mean = auc_mean / counter
print(f"K-Nearest Neighbors Average AUC: {auc_mean:.4f}")

# Predicting threads with Classification and Regression Trees
auc_mean = 0
counter = 0
for i in random_states:
    # 80/20 Train-test with degree distribution as predictors of the thread type
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, y, test_size=0.2, random_state=i)
    # Use training data to predict the probability of a thread being discussison based with a Decision Tree Classifier
    downstream_model = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    y_hat = downstream_model.predict_proba(X_test)[:, 1]
    # Evaluate model performance
    auc = roc_auc_score(y_test, y_hat)
    auc_mean += auc
    counter += 1
auc_mean = auc_mean / counter
print(f"Classification and Regression Trees Average AUC: {auc_mean:.4f}")

# Predicting threads with Naive Bayes
auc_mean = 0
counter = 0
for i in random_states:
    # 80/20 Train-test with degree distribution as predictors of the thread type
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, y, test_size=0.2, random_state=i)
    # Use training data to predict the probability of a thread being discussison based with Gaussian Naive Bayes
    downstream_model = GaussianNB().fit(X_train, y_train)
    y_hat = downstream_model.predict_proba(X_test)[:, 1]
    # Evaluate model performance
    auc = roc_auc_score(y_test, y_hat)
    auc_mean += auc
    counter += 1
auc_mean = auc_mean / counter
print(f"Naive Bayes Average AUC: {auc_mean:.4f}")

# Predicting threads with Support Vector Machines
auc_mean = 0
counter = 0
for i in random_states:
    # 80/20 Train-test with degree distribution as predictors of the thread type
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, y, test_size=0.2, random_state=i)
    # Use training data to predict the probability of a thread being discussison based with a support vector machine
    downstream_model = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    downstream_model = downstream_model.fit(X_train, y_train)
    y_hat = downstream_model.predict_proba(X_test)[:, 1]
    # Evaluate model performance
    auc = roc_auc_score(y_test, y_hat)
    auc_mean += auc
    counter += 1
auc_mean = auc_mean / counter
print(f"Support Vector Machines Average AUC: {auc_mean:.4f}")


# USING EMBEDDING VECTOR FOR THE WHOLE NETWORK AS THE FEATURE VECTOR
results = []
print("\n------------------------------------------------AUC Scores using karate club embeddings with LightGBM--------------------------------------------------------------\n")
# Fit a Feather model to the graphs
model = FeatherGraph()
model.fit(graphs)
# Get graph embedding
X_embeddings = model.get_embedding()

auc_mean = 0
counter = 0
for i in random_states:
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=i)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    # Params
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'is_unbalance': True,
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.04,
        'bagging_fraction': 0.95,
        'feature_fraction': 0.98,
        'bagging_freq': 6,
        'max_depth': -1,
        'max_bin': 511,
        'min_data_in_leaf': 20,
        'verbose': 0,
        'seed': i
    }
    downstream_model = lgb.train(params, lgb_train, num_boost_round=800, valid_sets=lgb_eval, early_stopping_rounds=20)
    y_hat = downstream_model.predict(X_test, num_iteration=downstream_model.best_iteration)
    auc = roc_auc_score(y_test, y_hat)
    auc_mean += auc
    counter += 1
auc_mean = auc_mean / counter
print('FeatherGraph Embeddings LightGBM average AUC: {:.4f}'.format(auc_mean))
results.append('FeatherGraph Embeddings LightGBM average AUC: {:.4f}'.format(auc_mean))

# Fit a FGSD model to the graphs
model = FGSD()
model.fit(graphs)
# Get graph embedding
X_embeddings = model.get_embedding()

auc_mean = 0
counter = 0
for i in random_states:
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=i)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    # Params
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'is_unbalance': True,
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.04,
        'bagging_fraction': 0.95,
        'feature_fraction': 0.98,
        'bagging_freq': 6,
        'max_depth': -1,
        'max_bin': 511,
        'min_data_in_leaf': 20,
        'verbose': 0,
        'seed': i
    }
    downstream_model = lgb.train(params, lgb_train, num_boost_round=800, valid_sets=lgb_eval, early_stopping_rounds=20)
    y_hat = downstream_model.predict(X_test, num_iteration=downstream_model.best_iteration)
    auc = roc_auc_score(y_test, y_hat)
    auc_mean += auc
    counter += 1
auc_mean = auc_mean / counter
print('FGSD Embeddings LightGBM average AUC: {:.4f}'.format(auc_mean))
results.append('FGSD Embeddings LightGBM average AUC: {:.4f}'.format(auc_mean))

# Fit a GeoScattering model to the graphs
model = GeoScattering()
model.fit(graphs)
# Get graph embedding
X_embeddings = model.get_embedding()

auc_mean = 0
counter = 0
for i in random_states:
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=i)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    # Params
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'is_unbalance': True,
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.04,
        'bagging_fraction': 0.95,
        'feature_fraction': 0.98,
        'bagging_freq': 6,
        'max_depth': -1,
        'max_bin': 511,
        'min_data_in_leaf': 20,
        'verbose': 0,
        'seed': i
    }
    downstream_model = lgb.train(params, lgb_train, num_boost_round=800, valid_sets=lgb_eval, early_stopping_rounds=20)
    y_hat = downstream_model.predict(X_test, num_iteration=downstream_model.best_iteration)
    auc = roc_auc_score(y_test, y_hat)
    auc_mean += auc
    counter += 1
auc_mean = auc_mean / counter
print('GeoScattering Embeddings LightGBM average AUC: {:.4f}'.format(auc_mean))
results.append('GeoScattering Embeddings LightGBM average AUC: {:.4f}'.format(auc_mean))

# Fit a GL2Vec model to the graphs
model = GL2Vec()
model.fit(graphs)
# Get graph embedding
X_embeddings = model.get_embedding()

auc_mean = 0
counter = 0
for i in random_states:
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=i)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    # Params
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'is_unbalance': True,
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.04,
        'bagging_fraction': 0.95,
        'feature_fraction': 0.98,
        'bagging_freq': 6,
        'max_depth': -1,
        'max_bin': 511,
        'min_data_in_leaf': 20,
        'verbose': 0,
        'seed': i
    }
    downstream_model = lgb.train(params, lgb_train, num_boost_round=800, valid_sets=lgb_eval, early_stopping_rounds=20)
    y_hat = downstream_model.predict(X_test, num_iteration=downstream_model.best_iteration)
    auc = roc_auc_score(y_test, y_hat)
    auc_mean += auc
    counter += 1
auc_mean = auc_mean / counter
print('GL2Vec Embeddings LightGBM AUC: {:.4f}'.format(auc_mean))
results.append('GL2Vec Embeddings LightGBM average AUC: {:.4f}'.format(auc_mean))

# Fit a Graph2Vec model to the graphs
model = Graph2Vec()
model.fit(graphs)
# Get graph embedding
X_embeddings = model.get_embedding()

auc_mean = 0
counter = 0
for i in random_states:
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=i)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    # Params
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'is_unbalance': True,
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.04,
        'bagging_fraction': 0.95,
        'feature_fraction': 0.98,
        'bagging_freq': 6,
        'max_depth': -1,
        'max_bin': 511,
        'min_data_in_leaf': 20,
        'verbose': 0,
        'seed': i
    }
    downstream_model = lgb.train(params, lgb_train, num_boost_round=800, valid_sets=lgb_eval, early_stopping_rounds=20)
    y_hat = downstream_model.predict(X_test, num_iteration=downstream_model.best_iteration)
    auc = roc_auc_score(y_test, y_hat)
    auc_mean += auc
    counter += 1
auc_mean = auc_mean / counter
print('Graph2Vec Embeddings LightGBM AUC: {:.4f}'.format(auc_mean))
results.append('Graph2Vec Embeddings LightGBM average AUC: {:.4f}'.format(auc_mean))

# # Fit a IGE model to the graphs
# model = IGE()
# model.fit(graphs)
# # Get graph embedding
# X_embeddings = model.get_embedding()

# auc_mean = 0
# counter = 0
# for i in random_states:
#     X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=i)
#     lgb_train = lgb.Dataset(X_train, y_train)
#     lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
#     # Params
#     params = {
#         'task': 'train',
#         'boosting_type': 'gbdt',
#         'objective': 'binary',
#         'is_unbalance': True,
#         'metric': 'binary_logloss',
#         'num_leaves': 31,
#         'learning_rate': 0.04,
#         'bagging_fraction': 0.95,
#         'feature_fraction': 0.98,
#         'bagging_freq': 6,
#         'max_depth': -1,
#         'max_bin': 511,
#         'min_data_in_leaf': 20,
#         'verbose': 0,
#         'seed': i
#     }
#     downstream_model = lgb.train(params, lgb_train, num_boost_round=800, valid_sets=lgb_eval, early_stopping_rounds=20)
#     y_hat = downstream_model.predict(X_test, num_iteration=downstream_model.best_iteration)
#     auc = roc_auc_score(y_test, y_hat)
#     auc_mean += auc
#     counter += 1
# auc_mean = auc_mean / counter
# print('IGE Embeddings Logistic Regression AUC: {:.4f}'.format(auc_mean))
# results.append('IGE Embeddings LightGBM average AUC: {:.4f}'.format(auc_mean))

# Fit a LDP model to the graphs
model = LDP()
model.fit(graphs)
# Get graph embedding
X_embeddings = model.get_embedding()

auc_mean = 0
counter = 0
for i in random_states:
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=i)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    # Params
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'is_unbalance': True,
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.04,
        'bagging_fraction': 0.95,
        'feature_fraction': 0.98,
        'bagging_freq': 6,
        'max_depth': -1,
        'max_bin': 511,
        'min_data_in_leaf': 20,
        'verbose': 0,
        'seed': i
    }
    downstream_model = lgb.train(params, lgb_train, num_boost_round=800, valid_sets=lgb_eval, early_stopping_rounds=20)
    y_hat = downstream_model.predict(X_test, num_iteration=downstream_model.best_iteration)
    auc = roc_auc_score(y_test, y_hat)
    auc_mean += auc
    counter += 1
auc_mean = auc_mean / counter
print('LDP Embeddings LightGBM AUC: {:.4f}'.format(auc_mean))
results.append('LDP Embeddings LightGBM average AUC: {:.4f}'.format(auc_mean))

# Fit a NetLSD model to the graphs
model = NetLSD()
model.fit(graphs)
# Get graph embedding
X_embeddings = model.get_embedding()

auc_mean = 0
counter = 0
for i in random_states:
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=i)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    # Params
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'is_unbalance': True,
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.04,
        'bagging_fraction': 0.95,
        'feature_fraction': 0.98,
        'bagging_freq': 6,
        'max_depth': -1,
        'max_bin': 511,
        'min_data_in_leaf': 20,
        'verbose': 0,
        'seed': i
    }
    downstream_model = lgb.train(params, lgb_train, num_boost_round=800, valid_sets=lgb_eval, early_stopping_rounds=20)
    y_hat = downstream_model.predict(X_test, num_iteration=downstream_model.best_iteration)
    auc = roc_auc_score(y_test, y_hat)
    auc_mean += auc
    counter += 1
auc_mean = auc_mean / counter
print('NetLSD Embeddings LightGBM AUC: {:.4f}'.format(auc_mean))
results.append('NetLSD Embeddings LightGBM average AUC: {:.4f}'.format(auc_mean))

# Fit a SF model to the graphs
model = SF()
model.fit(graphs)
# Get graph embedding
X_embeddings = model.get_embedding()

auc_mean = 0
counter = 0
for i in random_states:
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=i)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    # Params
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'is_unbalance': True,
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.04,
        'bagging_fraction': 0.95,
        'feature_fraction': 0.98,
        'bagging_freq': 6,
        'max_depth': -1,
        'max_bin': 511,
        'min_data_in_leaf': 20,
        'verbose': 0,
        'seed': i
    }
    downstream_model = lgb.train(params, lgb_train, num_boost_round=800, valid_sets=lgb_eval, early_stopping_rounds=20)
    y_hat = downstream_model.predict(X_test, num_iteration=downstream_model.best_iteration)
    auc = roc_auc_score(y_test, y_hat)
    auc_mean += auc
    counter += 1
auc_mean = auc_mean / counter
print('SF Embeddings LightGBM AUC: {:.4f}'.format(auc_mean))
results.append('SF Embeddings LightGBM average AUC: {:.4f}'.format(auc_mean))

# Fit a WaveletCharacteristic model to the graphs
model = WaveletCharacteristic()
model.fit(graphs)
# Get graph embedding
X_embeddings = model.get_embedding()

auc_mean = 0
counter = 0
for i in random_states:
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=i)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    # Params
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'is_unbalance': True,
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.04,
        'bagging_fraction': 0.95,
        'feature_fraction': 0.98,
        'bagging_freq': 6,
        'max_depth': -1,
        'max_bin': 511,
        'min_data_in_leaf': 20,
        'verbose': 0,
        'seed': i
    }
    downstream_model = lgb.train(params, lgb_train, num_boost_round=800, valid_sets=lgb_eval, early_stopping_rounds=20)
    y_hat = downstream_model.predict(X_test, num_iteration=downstream_model.best_iteration)
    auc = roc_auc_score(y_test, y_hat)
    auc_mean += auc
    counter += 1
auc_mean = auc_mean / counter
print('WaveletCharacteristic Embeddings LightGBM AUC: {:.4f}'.format(auc_mean))
results.append('WaveletCharacteristic Embeddings LightGBM average AUC: {:.4f}'.format(auc_mean))

print("\n\n")
for i in results:
    print(i)