import functools
import random
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

import graph as helper
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def mirror_pair(pairs, embeddedness_matrix, adj, existance_mat, marker):
    embeddedness_hist = Counter(embeddedness_matrix[tuple(pair)] for pair in pairs)
    # find possible mirror pairs, that should be a. of the same embeddedness_hist, b. unconnected, c. not already chosen
    candidates = zip(*np.where(
        functools.reduce(
            np.logical_and,
            (
                # has the same embeddness as the test data
                np.isin(embeddedness_matrix, list(embeddedness_hist.keys())),
                # not connected in the graph
                adj == 0,
                # both nodes exists in graph
                existance_mat == 1,
                # not already chosen
                marker == 0
            )
        )
    ))

    for candidate in candidates:
        if not embeddedness_hist:
            return

        candidate_embeddedness = embeddedness_matrix[candidate]

        if candidate_embeddedness in embeddedness_hist and embeddedness_hist[candidate_embeddedness] > 0:
            embeddedness_hist[candidate_embeddedness] -= 1
            marker[candidate] = 1
            yield candidate
        elif candidate_embeddedness in embeddedness_hist:
            del embeddedness_hist[candidate]


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
    logging.info("reading features")
    features = pd.read_csv(features_file, sep="\t")

    logging.info("preprocessing")
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

    logging.info("get positives")
    X, y = features.drop('edge_sign', axis=1), features['edge_sign']
    positives = list(zip(*np.where(adj > 0)))
    random.shuffle(positives)
    splitter = []

    for idx, test_chunk in enumerate(np.array_split(positives, 10)):
        logging.info(f"mirroring {idx + 1}")
        mirrors = mirror_pair(test_chunk, embeddedness_matrix, adj, existance_mat, marker)
        curr_test_vals = map(tuple, [*test_chunk, *mirrors])

        logging.info(f"slicing {idx + 1}")
        curr_test = features[features['Unnamed: 0'].isin([str(t) for t in curr_test_vals])]
        curr_train = features.drop(curr_test.index)
        splitter.append((list(curr_train.index), list(curr_test.index)))
    return X.drop(['Unnamed: 0'], axis=1), y, splitter


# def train_and_predict(train, test):
# logmodel = LogisticRegression(solver='sag', n_jobs=-1)
# logmodel.fit(X.loc[train], y.loc[train])
# predictions = logmodel.predict(X.loc[test])
# return (classification_report(y.loc[test], predictions))

def predict(features_file, tsv_file):
    data = pd.read_csv(features_file, sep="\t").rename(columns={'Unnamed: 0': 'K'})  # load data set
    data = data[(data.T != 0).any()]
    data = data.drop('K', axis=1)

    logging.info("splitting")
    X, y, splitter = split_data_balanced(features_file, tsv_file)

    labels = []
    preds = []
    for idx, (train, test) in enumerate(splitter):
        logging.info(f"classifying {idx + 1}")
        logmodel = XGBClassifier(n_jobs=10)
        logmodel.fit(X.loc[train], y.loc[train])
        predictions = logmodel.predict(X.loc[test])
        logging.info(f"postprocessing {idx + 1}")
        labels.extend(list(y.loc[test]))
        preds.extend(list(predictions))
    print(classification_report(labels, preds))


if __name__ == "__main__":
    logging.info("starting")
    predict('./calculated_features/features-1000-refactored.tsv', './datasets/wiki-demo-1000.tsv')

    # print("var1")
    # t1 = time_of_start_computation = datetime.now()
    #
    # with open('out-wiki-ml-varient1-maxiter.tsv', 'w') as f:
    #     print(predict('./calculated_features/wiki-full-features.tsv', './datasets/wiki.tsv'), file=f)
    #
    # t2 = datetime.now()
    # triads_time = t2 - t1
    # print(triads_time)
    #
    # print("var2")
    # t1 = time_of_start_computation = datetime.now()
    # with open('out-wiki-ml-varient2-maxiter.tsv', 'w') as f:
    #     print(predict('./calculated_features/wiki-varient2-full-features.tsv', './datasets/variant2-wiki.tsv'), file=f)
    #
    # t2 = datetime.now()
    # triads_time = t2 - t1
    # print(triads_time)
    #
    # print("var3")
    # t1 = time_of_start_computation = datetime.now()
    # with open('out-wiki-ml-varient3-maxiter.tsv', 'w') as f:
    #     print(predict('./calculated_features/wiki-varient3-full-features.tsv', './datasets/variant3-wiki.tsv'), file=f)
    #
    # t2 = datetime.now()
    # triads_time = t2 - t1
    # print(triads_time)
    #
    # print("var4")
    # t1 = time_of_start_computation = datetime.now()
    # with open('out-wiki-ml-varient4-maxiter.tsv', 'w') as f:
    #     print(predict('./calculated_features/wiki-varient4-full-features.tsv', './datasets/variant4-wiki.tsv'), file=f)
    #
    # t2 = datetime.now()
    # triads_time = t2 - t1
    # print(triads_time)
