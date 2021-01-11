import contextlib
import functools
import itertools
import random
from collections import Counter
import multiprocessing as mp
import re

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import lightgbm as lgbm

import graph as helper
import logging

logging.basicConfig(
    #filename='prediction_logs1',
    #filemode='a',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# This function returns a list of non-existing nodes with the same embeddedness in order to compile the 10 test chunks
def mirror_pair(pairs, embeddedness_matrix, adj, existance_mat, marker):
    # count all embeddedness values the the function should return
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

    # Iterate through all potential mirror pairs, until all mirror pairs are found
    for candidate in candidates:

        # All mirror pairs needed were found
        if not embeddedness_hist:
            return

        candidate_embeddedness = embeddedness_matrix[candidate]

        if candidate_embeddedness in embeddedness_hist and embeddedness_hist[candidate_embeddedness] > 0:
            embeddedness_hist[candidate_embeddedness] -= 1
            if embeddedness_hist[candidate_embeddedness] == 0:
                del embeddedness_hist[candidate]

            marker[candidate] = 1
            yield candidate


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
    logger.info("reading features")
    features = pd.read_csv(features_file, sep="\t").rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    logger.info("preprocessing")
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

    logger.info("get positives")
    X, y = features.drop('edge_sign', axis=1), features['edge_sign']
    positives = list(zip(*np.where(adj > 0)))
    random.shuffle(positives)
    splitter = []

    blabla = []
    for idx, test_chunk in enumerate(np.array_split(positives, 10)):
        logger.info(f"mirroring {idx + 1}")
        mirrors = mirror_pair(test_chunk, embeddedness_matrix, adj, existance_mat, marker)
        curr_test_vals = list(map(tuple, [*test_chunk, *mirrors]))

        blabla.append(curr_test_vals)

    flattened = list(itertools.chain(*blabla))
    balanced_df = features[features['K'].isin([str(t) for t in flattened])]

    for idx, b in enumerate(blabla):
        logger.info(f"feeding {idx + 1}")
        curr_test = balanced_df[balanced_df['K'].isin([str(t) for t in b])]
        curr_train = balanced_df.drop(curr_test.index)
        splitter.append((list(curr_train.index), list(curr_test.index)))
    return X.drop(['K'], axis=1), y, splitter


def predict_splice(train, test, X, y, idx):


    logger.info(f"classifying {idx + 1}")

    logmodel = lgbm.LGBMClassifier()
    #logmodel = XGBClassifier()
    logmodel.fit(X.loc[train], y.loc[train])
    prediction = logmodel.predict(X.loc[test])
    with open(f'C:/Users/omyiz/Documents/repos/positive_edges_prediction/xxxxxx.txt', 'w') as xx:
        xx.write(classification_report(y.loc[test], prediction))
    logger.info(f"postprocessing {idx + 1} XXXXXXXX")
    return y.loc[test], prediction, test


def predict(features_file, tsv_file):
    data = pd.read_csv(features_file, sep="\t").rename(columns={'Unnamed: 0': 'K'})
    #data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        # .drop(['positive_in_degree(u)',
        #                 'positive_out_degree(u)', 'negative_in_degree(u)', 'negative_out_degree(u)',
        #                 'positive_in_degree(v)', 'positive_out_degree(v)', 'negative_in_degree(v)',
        #                 'negative_out_degree(v)', 'total_out_degree(u)',
        #                 'total_in_degree(v)', 'C(u,v)', 'edge_sign'], axis=1)  # load data set
    data = data[(data.T != 0).any()]
    #data = data.drop('K', axis=1)

    logger.info("splitting")
    X, y, splitter = split_data_balanced(features_file, tsv_file)

    labels = []
    preds = []
    # pool = mp.Pool(10)
    # results = [pool.apply(predict_splice, args=(train, test, X, y, idx)) for idx, (train, test) in enumerate(splitter)]
    # pool.close()
    #
    # for label, predictions, test in results:
    #     labels.extend(list(y.loc[test]))
    #     preds.extend(list(predictions))
    #
    # print(classification_report(labels, preds))
    for idx, (train, test) in enumerate(splitter):
        logger.info(f"classifying {idx + 1}")
        logmodel = lgbm.LGBMClassifier()
        #logmodel = XGBClassifier(n_jobs=20)
        logmodel.fit(X.loc[train], y.loc[train])
        predictions = logmodel.predict(X.loc[test])
        logger.info(f"postprocessing {idx + 1}")
        logger.info(classification_report(list(y.loc[test]), list(predictions)))
        labels.extend(list(y.loc[test]))
        preds.extend(list(predictions))

    return classification_report(labels, preds)


@contextlib.contextmanager
def timer():
    import time

    start = time.time()
    yield
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    logger = logging.getLogger("logs")
    logger.info("starting")

    with open('100-predict.tsv', 'w') as f:
        print(predict('./calculated_features/assertion-normalize-1000-float-round.tsv', './datasets/wiki-demo-1000.tsv'), file=f)

    logger.info("starting v1")

    with open('out-wiki-ml-variant1.tsv', 'w') as f:
        print(predict('./calculated_features_treshold/var1-features-treshold.tsv', './datasets/variant1-wiki.tsv'), file=f)

    logger.info("starting v2")
    with open('out-wiki-ml-variant2.tsv', 'w') as f:
        print(predict('./calculated_features_treshold/var2-features-treshold.tsv', './datasets/variant2-wiki.tsv'), file=f)

    logger.info("starting v3")
    with open('out-wiki-ml-variant3.tsv', 'w') as f:
        print(predict('./calculated_features_treshold/var3-features-treshold.tsv', './datasets/variant3-wiki.tsv'), file=f)

    logger.info("starting v4")
    with open('out-wiki-ml-variant4.tsv', 'w') as f:
        print(predict('./calculated_features_treshold/var4-features-treshold.tsv', './datasets/variant4-wiki.tsv'), file=f)
