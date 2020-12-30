import pandas as pd
import networkx as nx
import numpy as np
import multiprocessing as mp
import random
import functools
import graph as helper

from scipy import sparse
from datetime import datetime

from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def mirror_pair(pair, embeddedness_matrix, adj, existance_mat, marker):
    embeddedness = embeddedness_matrix[pair]
    # find possible mirror pairs, that should be a. of the same embeddedness, b. unconnected, c. not already chosen
    candidates = zip(*np.where(
        functools.reduce(np.logical_and,
                         (embeddedness_matrix == embeddedness, adj == 0, existance_mat == 1, marker == 0)))
    )

    candidate = next(candidates, None)
    if candidate:
        marker[candidate] = 1
    return candidate


def calc_existence_mat(nodes):
    _zs = np.zeros(max(nodes) + 1, dtype=int)

    for i in nodes:
        _zs[i] = 1
    _zs = _zs[:, np.newaxis]

    # the result of this matrix mul will yield 1 only in cells [i, j]
    # where both Node i and j are ones, meaning, they both exists in the graph
    existence_mat = _zs * _zs.T

    # Since we don't need the diagonal, we can zero it.
    np.fill_diagonal(existence_mat, 0)
    return existence_mat


def split_data_balanced(features_file, tsv_file):
    features = pd.read_csv(features_file, sep="\t")

    # calculate adj matrix
    G, real_nodes = helper.build_graph(tsv_file, record_fake_data=True)

    A = sparse.csr_matrix(nx.to_numpy_matrix(G, weight="weight"))

    # Matrix M[i,j] showing adjacency between Node i and j
    adj = A.todense()

    A[A != 0] = 1
    B = A + A.T
    # Matrix M[i,j] showing number of common neighbors between Node i and j
    embeddedness_matrix = (B @ B.T).todense()
    # Marks the nonedges already chosen
    marker = np.zeros(A.shape)
    # Matrix M[i,j] marks whether both Nodes i and j exists in the graph
    existance_mat = calc_existence_mat(real_nodes)

    pos = list(zip(*np.where(adj > 0)))
    random.shuffle(pos)
    test_pos = pos[:len(pos) // 10]
    mirrors = filter(lambda p: p is not None,
                     [mirror_pair(p, embeddedness_matrix, adj, existance_mat, marker) for p in test_pos])

    # features = features.set_index('Unnamed: 0')
    test = [*test_pos, *mirrors]
    test_df = features[features['Unnamed: 0'].isin([str(t) for t in test])]
    train_df = features.drop(test_df.index, errors='ignore')

    test_df = test_df.drop(['Unnamed: 0'], axis=1)
    train_df = train_df.drop(['Unnamed: 0'], axis=1)

    X_train, y_train = train_df.drop('edge_sign', axis=1), train_df['edge_sign']
    X_test, y_test = test_df.drop('edge_sign', axis=1), test_df['edge_sign']

    return X_train, X_test, y_train, y_test


def predict(features_file, tsv_file):
    data = pd.read_csv(features_file, sep="\t").rename(columns={'Unnamed: 0': 'K'})  # load data set
    data = data[(data.T != 0).any()]
    data = data.drop('K', axis=1)

    X_train, X_test, y_train, y_test = split_data_balanced(features_file, tsv_file)
    logmodel = LogisticRegression(n_jobs=mp.cpu_count())
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)
    return classification_report(y_test, predictions)


if __name__ == "__main__":
    print("K")
    print(predict('./features-100-refactored.tsv', './datasets/wiki-demo-100.tsv'))

    # print("var1")
    # t1 = time_of_start_computation = datetime.now()
    #
    # with open('out-wiki-ml-varient1.tsv', 'w') as f:
    #     print(predict('./calculated_features/wiki-full-features.tsv'), file=f)
    #
    # t2 = datetime.now()
    # triads_time = t2 - t1
    # print(triads_time)
    #
    # print("var2")
    # t1 = time_of_start_computation = datetime.now()
    # with open('out-wiki-ml-varient2.tsv', 'w') as f:
    #     print(predict('./calculated_features/wiki-varient2-full-features.tsv', ), file=f)
    #
    # t2 = datetime.now()
    # triads_time = t2 - t1
    # print(triads_time)
    #
    # print("var3")
    # t1 = time_of_start_computation = datetime.now()
    # with open('out-wiki-ml-varient3.tsv', 'w') as f:
    #     print(predict('./calculated_features/wiki-varient3-full-features.tsv'), file=f)
    #
    # t2 = datetime.now()
    # triads_time = t2 - t1
    # print(triads_time)
