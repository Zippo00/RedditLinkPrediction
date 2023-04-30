import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from karateclub.dataset import GraphSetReader
from karateclub import FeatherGraph
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Initialize graph reader
reader = GraphSetReader("reddit10k")

# Reddit thread data
graphs = reader.get_graphs()
y = reader.get_target()

# 10 Train-test random state values for deterministic set selections
random_states = [0, 1, 5, 12, 38, 42, 47, 49, 72, 77]
# List of feature vectors for each node (Degree Distribution as feature vector)
# Feature vector = (Amount of nodes with degree of 1, Amount of nodes with degree of 2, Amount of nodes with degree of 3, Amount of nodes with degree of 4, Amount of nodes with degree of 5+)
feature_vectors = []
for i in graphs:
    feature_vector = [0, 0, 0, 0, 0]
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
        if j > 4:
            feature_vector[4] += deg_counts[i]
    #print(f"Feature vector: {feature_vector}")
    feature_vectors.append(feature_vector)

    #                                                           **UNCOMMENT CODE BELOW TO PLOT THE GRAPH AND DEGREE DISTRIBUTION FOR EACH REDDIT THREAD**
    # fig = plt.figure("Degree of a graph", figsize=(8, 8))
    # axgrid = fig.add_gridspec(5, 4)

    # ax0 = fig.add_subplot(axgrid[0:3, :])
    # Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # pos = nx.spring_layout(Gcc, seed=10396953)
    # nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    # nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    # ax0.set_title("Connected components of G")
    # ax0.set_axis_off()

    # ax2 = fig.add_subplot(axgrid[3:, :])
    # ax2.bar(*np.unique(degree_sequence, return_counts=True))
    # ax2.set_title("Degree histogram")
    # ax2.set_xlabel("Degree")
    # ax2.set_ylabel("# of Nodes")

    # fig.tight_layout()
    # plt.show()


# USING EMBEDDING VECTOR FOR THE WHOLE NETWORK AS THE FEATURE VECTOR
# Fit a Feather model to the graphs
model = FeatherGraph()
model.fit(graphs)
# Get graph embedding
X_embedding = model.get_embedding()




# Predicting threads with Logistic Regression
auc_mean = 0
counter = 0
for i in random_states:
    # 80/20 Train-test with degree distribution as predictors of the thread type
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, y, test_size=0.2, random_state=i)
    # Use training data to predict the probability of a thread being discusison based with a logistic regression model
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
    # Use training data to predict the probability of a thread being discusison based with a logistic regression model
    downstream_model = LinearDiscriminantAnalysis(random_state=0, max_iter=1000).fit(X_train, y_train)
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
    # Use training data to predict the probability of a thread being discusison based with a logistic regression model
    downstream_model = KNeighborsClassifier(random_state=0, max_iter=1000).fit(X_train, y_train)
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
    # Use training data to predict the probability of a thread being discusison based with a logistic regression model
    downstream_model = DecisionTreeClassifier(random_state=0, max_iter=1000).fit(X_train, y_train)
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
    # Use training data to predict the probability of a thread being discusison based with a logistic regression model
    downstream_model = GaussianNB(random_state=0, max_iter=1000).fit(X_train, y_train)
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
    # Use training data to predict the probability of a thread being discusison based with a logistic regression model
    downstream_model = SVC(random_state=0, max_iter=1000).fit(X_train, y_train)
    y_hat = downstream_model.predict_proba(X_test)[:, 1]
    # Evaluate model performance
    auc = roc_auc_score(y_test, y_hat)
    auc_mean += auc
    counter += 1
auc_mean = auc_mean / counter
print(f"Support Vector Machines Average AUC: {auc_mean:.4f}")
