import functools
import itertools
import random
from collections import Counter
import re

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

import graph as helper
import logging

from features import FEATURES_PATH
from graph import DATASET_PATH
PREDICTIONS_PATH = "./predictions/"


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# This function returns a list of non-existing nodes with the same embeddedness in order to compile the 10 balanced chunks
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

        # Add relevant candidates and make sure they will not be chosen again using the marker
        if candidate_embeddedness in embeddedness_hist and embeddedness_hist[candidate_embeddedness] > 0:
            embeddedness_hist[candidate_embeddedness] -= 1
            if embeddedness_hist[candidate_embeddedness] == 0:
                del embeddedness_hist[candidate]

            marker[candidate] = 1
            yield candidate


def split_data_balanced(features_file, tsv_file):
    logger.info("reading features")
    features = pd.read_csv(features_file, sep="\t")
    features.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x), inplace=True)
    features.rename(columns={features.columns[0]: 'K'}, inplace=True)
    """TODO: copy first part from the remote"""

    logger.info("preprocessing")
    # calculate adj matrix: matrix M[i,j] showing adjacency between Node i and j
    G, real_nodes = helper.build_graph(tsv_file, record_fake_data=True)
    A = sparse.csr_matrix(nx.to_numpy_matrix(G, weight="weight"))
    adj = A.todense()

    # Matrix M[i,j] counts the number of common neighbors for nodes i, j in the graph
    embeddedness_matrix = helper.get_embeddedbess_matrix(G)

    # Marks the nonedges already chosen
    marker = np.zeros(A.shape)

    # Matrix M[i,j] marks whether both Nodes i and j exists in the graph
    existance_mat = helper.calc_existence_mat(real_nodes)

    logger.info("get positives")
    X, y = features.drop('edge_sign', axis=1), features['edge_sign']
    positives = list(zip(*np.where(adj > 0)))
    random.shuffle(positives)
    splitter = []
    all_test_chunks = []

    # Create balanced 10 chunks for 10 fold. Data not in any of the chunks will not be used as train nor test data
    for idx, test_chunk in enumerate(np.array_split(positives, 10)):
        logger.info(f"mirroring {idx + 1}")
        mirrors = mirror_pair(test_chunk, embeddedness_matrix, adj, existance_mat, marker)
        curr_test_vals = list(map(tuple, [*test_chunk, *mirrors]))
        all_test_chunks.append(curr_test_vals)

    flattened = list(itertools.chain(*all_test_chunks))
    balanced_df = features[features['K'].isin([str(t) for t in flattened])]

    # Create indexes for 10 train-test chunks, according to k-fold
    for idx, test_chunck in enumerate(all_test_chunks):
        logger.info(f"feeding {idx + 1}")
        curr_test = balanced_df[balanced_df['K'].isin([str(t) for t in test_chunck])]
        curr_train = balanced_df.drop(curr_test.index)
        splitter.append((list(curr_train.index), list(curr_test.index)))

    return X.drop(['K'], axis=1), y, splitter


def predict(features_file, tsv_file):
    data = pd.read_csv(features_file, sep="\t")
    # Remove empty lines, if exists
    data = data[(data.T != 0).any()]

    logger.info("splitting")
    X, y, splitter = split_data_balanced(features_file, tsv_file)

    labels = []
    preds = []

    # Predict for each 1 out of 10 chunks: print classification report for each chunk to file (for debug purposes)
    for idx, (train, test) in enumerate(splitter):
        logger.info(f"classifying {idx + 1}")
        logmodel = LogisticRegression()
        logmodel.fit(X.loc[train], y.loc[train])
        predictions = logmodel.predict(X.loc[test])
        logger.info(f"postprocessing {idx + 1}")
        logger.info(classification_report(list(y.loc[test]), list(predictions)))
        labels.extend(list(y.loc[test]))
        preds.extend(list(predictions))

    # Return classification report based on the performance on 10 chunks
    return classification_report(labels, preds)



if __name__ == "__main__":
    logger = logging.getLogger()

    logger.info("starting v1")
    with open(PREDICTIONS_PATH + 'wiki-ml-variant1.txt', 'w') as f:
        print(predict(FEATURES_PATH + 'var1-features.tsv', DATASET_PATH + 'variant1-wiki.tsv'), file=f)

    logger.info("starting v2")
    with open(PREDICTIONS_PATH + 'wiki-ml-variant2.txt', 'w') as f:
        print(predict(FEATURES_PATH + 'var2-features.tsv', DATASET_PATH + 'variant2-wiki.tsv'), file=f)

    logger.info("starting v3")
    with open(PREDICTIONS_PATH + 'wiki-ml-variant3.txt', 'w') as f:
        print(predict(FEATURES_PATH + 'var3-features.tsv', DATASET_PATH + 'variant3-wiki.tsv'), file=f)

    logger.info("starting v4")
    with open(PREDICTIONS_PATH + 'wiki-ml-variant4.txt', 'w') as f:
        print(predict(FEATURES_PATH + 'var4-features.tsv', DATASET_PATH + 'variant4-wiki.tsv'), file=f)
