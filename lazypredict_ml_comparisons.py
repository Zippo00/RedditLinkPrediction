'''
Python file for computing machine learning comparisons with lazypredict library

Author: Mikko Lempinen
'''

import pandas as pd
import numpy as np
import networkx as nx
from karateclub.dataset import GraphSetReader
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


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

#Test data
#data = load_breast_cancer()
#feature_vectors = data.data
#y= data.target

# Generate feature vectors for each graph (Uncomment one of these to use)
#feature_vectors, index = deg_feature_vectors(graphs), 0
#feature_vectors, index = betweenness_centrality_vectors(graphs), 1
feature_vectors, index = closeness_centrality_vectors(graphs), 2


feature_vectors = np.array(feature_vectors)
count = 0
for j, i in enumerate(random_states):
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, y, test_size=0.2, random_state=i)
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models_train, predictions_train = clf.fit(X_train, X_train, y_train, y_train)
    models_test, predictions_test = clf.fit(X_train, X_test, y_train, y_test)
    count += 1
    if j == 0:
        AUCs = models_test.copy()
        del AUCs['Accuracy']
        del AUCs['Balanced Accuracy']
        del AUCs['F1 Score']
        del AUCs['Time Taken']
        continue 
    for index, row in AUCs.iterrows():
        for index2, row2 in models_test.iterrows():
            if index == index2:
                AUCs.at[index, 'ROC AUC'] += models_test.at[index2, 'ROC AUC']
    #print(f"AUCs: {AUCs}\n\n")
    #print(f"models_test: {models_test}\n\n")
    # if count > 2:
    #     break
for index in AUCs.index:
    auc_sum = AUCs['ROC AUC'][index]
    AUCs.at[index, 'ROC AUC'] = (auc_sum / count)

print(f"Average AUC scores:\n{AUCs}")


