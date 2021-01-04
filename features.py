import pandas as pd
import graph as helper
import networkx as nx
import itertools
import numpy as np
import multiprocessing as mp
from scipy import sparse
from tqdm import tqdm
from datetime import datetime

import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

"""Each triad type is represented by four digits as each digit may be 0 or 1.
Let w be the common neighbor of two nodes in a triad, u and v:
The first digit represent the direction of the edge between u,w: 0 if u points to w, else 1.
The second digit represents the sign of the edge between u,w: 0 if it`s positive, else 1.
The third digit represents the direction of the edge between v,w: 0 if v points to w, else 1.
The fourth digit represents the sign of the edge between v,w: 0 if it`s positive, else 1.
"""
TRIADS_TYPES = [(0, 0, 0, 0),
                (0, 0, 0, 1),
                (0, 0, 1, 0),
                (0, 0, 1, 1),
                (0, 1, 0, 0),
                (0, 1, 0, 1),
                (0, 1, 1, 0),
                (0, 1, 1, 1),
                (1, 0, 0, 0),
                (1, 0, 0, 1),
                (1, 0, 1, 0),
                (1, 0, 1, 1),
                (1, 1, 0, 0),
                (1, 1, 0, 1),
                (1, 1, 1, 0),
                (1, 1, 1, 1)]


NUMBER_OF_CORES = mp.cpu_count()


class Features_calculator:
    def __init__(self):
        self.features_df = None
        self.graph = None
        self.matrix = None
        self.undirected_graph = None
        self.embeddedness_matrix = None

    @staticmethod
    def build_features_df(graph):
        triads_data = {str(triad): 0 for triad in TRIADS_TYPES}
        degrees_data = {'positive_in_degree(u)': [0], 'positive_out_degree(u)': [0], 'negative_in_degree(u)': [0],
                        'negative_out_degree(u)': [0], 'positive_in_degree(v)': [0], 'positive_out_degree(v)': [0],
                        'negative_in_degree(v)': [0], 'negative_out_degree(v)': [0], 'total_out_degree(u)': [0],
                        'total_in_degree(v)': [0], 'C(u,v)': [0], 'edge_sign': [0]}
        triads_data.update(degrees_data)
        triads_index = []
        nodes_with_data = set(itertools.chain(*list(graph.edges)))
        for (u, v) in itertools.combinations(nodes_with_data, 2):
            triads_index.extend([(u, v), (v, u)])

        features_df = pd.DataFrame(triads_data, triads_index)
        return features_df

    @staticmethod
    def get_triad_status(u, w, v, triad, matrix):
        first_edge = 0
        second_edge = 0

        sign1 = -1 if triad[1] else 1
        sign2 = -1 if triad[3] else 1

        if (not triad[0]):
            first_edge = 1 if matrix[u, w] == sign1 else 0
        else:
            first_edge = 1 if matrix[w, u] == sign1 else 0

        if (not triad[2]):
            second_edge = 1 if matrix[v, w] == sign2 else 0

        else:
            second_edge = 1 if matrix[w, v] == sign2 else 0

        return 1 if (first_edge and second_edge) else 0


    def process_frame(self, features_df):
        for (u, v), row in features_df.iterrows():
            if self.matrix[u, v] != 0:
                features_df.at[(u, v), 'edge_sign'] = self.matrix[u, v]

            triads_dict_for_pair = {triad: 0 for triad in TRIADS_TYPES}
            for w in sorted(nx.common_neighbors(self.undirected_graph, u, v)):
                for triad in TRIADS_TYPES:
                    triad_status = self.get_triad_status(u, w, v, triad, self.matrix)
                    if triad_status:
                        triads_dict_for_pair[triad] += 1
            for triad in triads_dict_for_pair.keys():
                features_df.at[(u, v), str(triad)] = triads_dict_for_pair[triad]

            count = 0
            for node in (u, v):
                node_id = int(node)

                features_df.at[(u, v), 'positive_in_degree({})'.format('v' if count else 'u')] = self.positive_in[node_id]
                features_df.at[(u, v), 'positive_out_degree({})'.format('v' if count else 'u')] = self.positive_out[node_id]
                features_df.at[(u, v), 'negative_in_degree({})'.format('v' if count else 'u')] = self.negative_in[node_id]
                features_df.at[(u, v), 'negative_out_degree({})'.format('v' if count else 'u')] = self.negative_out[node_id]

                count += 1

            features_df.at[(u, v), 'total_out_degree(u)'] = self.negative_out[int(u)] + self.positive_out[int(u)]
            features_df.at[(u, v), 'total_in_degree(v)'] = self.negative_in[int(v)] + self.positive_in[int(v)]
            features_df.at[(u, v), 'C(u,v)'] = self.embeddedness_matrix[int(u), int(v)]



        return features_df

    def compute_features(self, tsv_file):
        graph = helper.build_graph(tsv_file)
        self.graph = graph
        self.undirected_graph = graph.to_undirected()
        features_df = self.build_features_df(graph)
        self.matrix = nx.to_numpy_matrix(graph, weight="weight")

        A = sparse.csr_matrix(nx.to_numpy_matrix(graph, weight="weight"))
        adj = A.todense()
        A[A != 0] = 1
        B = A + A.T
        self.embeddedness_matrix = (B @ B.T).todense()

        # demo < 0 -> positive reviews.  demo > 0 -> negative reviews
        # axis = 0 -> incoming.          axis = 1 -> outgoing
        self.positive_in = np.where(self.matrix < 0, 0, self.matrix).sum(axis=0)
        self.positive_out = np.where(self.matrix < 0, 0, self.matrix).sum(axis=1)
        self.negative_in = -1*np.where(self.matrix > 0, 0, self.matrix).sum(axis=0)
        self.negative_out = -1*np.where(self.matrix > 0, 0, self.matrix).sum(axis=1)

        features_df_split = np.array_split(features_df, NUMBER_OF_CORES)
        pool = mp.Pool(NUMBER_OF_CORES)
        parts = pool.map(self.process_frame, features_df_split)
        df = pd.concat(parts)
        # df = self.process_frame(features_df)
        pool.close()
        pool.join()
        return df


if __name__ == '__main__':

    td = Features_calculator().compute_features("./datasets/wiki-demo-100.tsv")
    td.to_csv("assertion.tsv", sep="\t")

    logging.info("V1")
    td = Features_calculator().compute_features("./datasets/variant1-wiki.tsv")
    td.to_csv("var1-features.tsv", sep="\t")

    logging.info("V2")
    td = Features_calculator().compute_features("./datasets/variant2-wiki.tsv")
    td.to_csv("var2-features.tsv", sep="\t")

    logging.info("V3")
    td = Features_calculator().compute_features("./datasets/variant3-wiki.tsv")
    td.to_csv("var3-features.tsv", sep="\t")

    logging.info("V4")
    td = Features_calculator().compute_features("./datasets/variant4-wiki.tsv")
    td.to_csv("var4-features.tsv", sep="\t")

