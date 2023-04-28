from karateclub.dataset import GraphSetReader
from karateclub import FeatherGraph
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# Initialize graph reader
reader = GraphSetReader("reddit10k")

# Reddit thread data
graphs = reader.get_graphs()
y = reader.get_target()

# Fit a Feather model to the graphs
model = FeatherGraph()
model.fit(graphs)
# Get graph embedding
X = model.get_embedding()

# 80/20 Train-test with graph embedding features as predictors of the thread type
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use training data to predict the probability of a thread being discusison based with a logistic regression model
downstream_model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
y_hat = downstream_model.predict_proba(X_test)[:, 1]
# Evaluate model performance
auc = roc_auc_score(y_test, y_hat)
print('AUC: {:.4f}'.format(auc))