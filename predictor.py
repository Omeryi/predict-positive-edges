import pandas as pd
import networkx as nx
import numpy as np
import multiprocessing as mp
import random
import functools

from datetime import datetime

from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def mirror_pair(pair, embeddedness_matrix, adj, marker):
    embeddedness = embeddedness_matrix[pair]
    equal_pair = next(zip(*np.where(functools.reduce(np.logical_and, (embeddedness_matrix == embeddedness, adj == 0, marker == 0)))))
    if equal_pair is not None:
        marker[equal_pair] = 1
    return equal_pair


def split_data_balanced(features, tsv_file):
    # calculate adj matrix
    G = helper.build_graph(tsv_file)
    A = sparse.csr_matrix(nx.to_numpy_matrix(G, weight="weight"))

    # Matrix M[i,j] showing adjacency between Node i and j
    adj = A.todense()

    A[A != 0] = 1
    B = A + A.T
    # Matrix M[i,j] showing number of common neighbors between Node i and j
    embeddedness_matrix = (B @ B.T).todense()
    # Marks the nonedges already chosen
    marker = np.zeros(A.shape)

    pos = list(zip(*np.where(adj > 0)))
    random.shuffle(pos)
    test_pos = pos[:len(pos) // 10]
    mirrors = [mirror_pair(p, embeddedness_matrix, adj, marker) for p in test_pos]

    test = [*test_pos, *mirrors]
    test_df = features.reindex([str(t) for t in test]).fillna(0)
    train_df = features.drop(test_df.index, errors='ignore')

    X_train, y_train = train_df.drop('edge_sign', axis=1), train_df['edge_sign']
    X_test, y_test = test_df.drop('edge_sign', axis=1), test_df['edge_sign']

    return X_train, y_train, X_test, y_test


def predict(features_file, tsv_file):
    data = pd.read_csv(features_file, sep="\t").rename(columns={'Unnamed: 0': 'K'})  # load data set
    data = data[(data.T != 0).any()]
    data = data.drop('K', axis=1)

    X_train, X_test, Y_train, Y_test = split_data_balance(features_file, tsv_file)
    logmodel = LogisticRegression(n_jobs=mp.cpu_count())
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)
    return classification_report(y_test, predictions)

if __name__ == "__main__":
    print("var1")
    t1 = time_of_start_computation = datetime.now()

    with open('out-wiki-ml-varient1.tsv', 'w') as f:
        print(predict('./calculated_features/wiki-full-features.tsv'), file=f)

    t2 = datetime.now()
    triads_time = t2 - t1
    print(triads_time)


    print("var2")
    t1 = time_of_start_computation = datetime.now()
    with open('out-wiki-ml-varient2.tsv', 'w') as f:
        print(predict('./calculated_features/wiki-varient2-full-features.tsv',), file=f)

    t2 = datetime.now()
    triads_time = t2 - t1
    print(triads_time)

    print("var3")
    t1 = time_of_start_computation = datetime.now()
    with open('out-wiki-ml-varient3.tsv', 'w') as f:
        print(predict('./calculated_features/wiki-varient3-full-features.tsv'), file=f)

    t2 = datetime.now()
    triads_time = t2 - t1
    print(triads_time)

