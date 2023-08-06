import numpy as np
import networkx as nx
import random
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_distances



class VariableKNN(BaseEstimator):

    def __init__(self):
        self.graph = None

    def fit(self, X, y, K=None, A=None, S=None):
        """
        Fit the VariableKNN model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        K : array-like of shape (n_samples,), optional
            The sequence of k-neighbors for each sample, by default None.
        A : array-like of shape (n_samples, n_samples), optional
            The adjacency matrix representing the graph, by default None.
        S : array-like of shape (n_samples, n_samples), optional
            The similarity matrix representing the graph, by default None.
        """
        self.X = X
        self.weighted_graph = False

        if K is not None:
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from([(_, dict(label=y[_])) for _ in range(len(X))])
            for i in range(len(X)):
                k_i = K[i]
                distances = np.linalg.norm(X - X[i], axis=1)
                nearest_indices = np.argsort(distances)[1:k_i+1]
                for j in nearest_indices:
                    self.graph.add_edge(i, j)

        elif A is not None:
            is_symmetric = np.allclose(A, A.T)
            if is_symmetric:
                self.graph = nx.from_numpy_matrix(A)
            else:
                self.graph = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
            labels = {i: y[i] for i in range(len(X))}
            nx.set_node_attributes(self.graph, labels, 'label')
        
        elif S is not None:
            self.weighted_graph = True
            num_nodes = S.shape[0]
            self.graph = nx.Graph()
            self.graph.add_nodes_from([(_, dict(label=y[_])) for _ in range(len(X))])
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    weight = S[i, j]
                    if weight != 0:
                        self.graph.add_edge(i, j, weight=weight)



    def predict(self, X):
        """
        Predict the class labels for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.X)
        X_closest = nbrs.kneighbors(X, return_distance=False)    # `X_closest` contains the indices of the closest data point in the training data(i.e., self.X) to each data point in the test data(i.e., X), which is also the node representation in self.graph
        if not self.weighted_graph:
            y_pred = propogation_algorighm(graph=self.graph, X_closest=X_closest, method='mode')
        else:
            y_pred = propogation_algorighm(graph=self.graph, X_closest=X_closest, method='weighted_mode')
        return y_pred
    

def construct_adjacency_matrix(X, k):
    # Compute pairwise distances between samples
    distances = pairwise_distances(X)
    
    # Sort distances and select top k nearest neighbors
    indices = np.argsort(distances, axis=1)[:, 1:k+1]
    
    # Construct adjacency matrix
    A = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        A[i, indices[i]] = 1
    
    return A

def propogation_algorighm(graph, X_closest, method='mode')-> np.ndarray:
    """

    Parameters
    ----------
    X_closest : np.ndarray of shape (n_queries, 1)
        the closest neighbor of each point-to-predict in the graph

    Returns
    -------
    np.ndarray
        y_pred
    """
    y_pred = []
    for row in range(X_closest.shape[0]):
        node_closest = X_closest[row][0]
        nbrs_closest = [node_closest]    # include `node_closest` itself
        nbrs_closest.extend(list(graph[node_closest].keys()))
        if method.lower() == 'mode':
            labels = [graph.nodes[nbr]['label'] for nbr in nbrs_closest]
            
            # I want all tie to be settled by the closest neighbor.
            closest_label = graph.nodes[node_closest]['label']
            max_count = max(labels.count(label) for label in labels)
            max_labels = [label for label in labels if labels.count(label) == max_count]
            if closest_label in max_labels:    # I want all tie to be settled by the closest neighbor.
                y_pred.append(closest_label)
            else:
                y_pred.append(random.choice(max_labels))
            
        elif method.lower() == 'weighted_mode':
            labels = []
            for nbr in nbrs_closest:
                if node_closest != nbr:
                    labels.append((graph.nodes[nbr]['label'], graph.get_edge_data(node_closest, nbr)['weight']))
                else:
                    labels.append((graph.nodes[node_closest]['label'], 1.))
            total_weights = defaultdict(float)
            for k, v in labels:
                total_weights[k] += v
            y_pred.append(max(total_weights, key=lambda x: total_weights[x]))
    return np.array(y_pred)