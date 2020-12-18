import pandas as pd
import graph as helper
import networkx as nx
import itertools
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from datetime import datetime

# Each triad type is represented by four digits as each digit may be 0 or 1.
# Let w be the common neighbor of two nodes in a triad, u and v:
# The first digit represent the direction of the edge between u,w: 0 if u points to w, else 1.
# The second digit represents the sign of the edge between u,w: 0 if it`s positive, else 1.
# The third digit represents the direction of the edge between v,w: 0 if v points to w, else 1.
# The fourth digit represents the sign of the edge between v,w: 0 if it`s positive, else 1.

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

# currently redundent
REVERSE_TRIADS = {(0, 0, 0, 0): (0, 0, 0, 0),
                  (0, 0, 0, 1): (0, 1, 0, 0),
                  (0, 0, 1, 0): (1, 0, 0, 0),
                  (0, 0, 1, 1): (1, 1, 0, 0),
                  (0, 1, 0, 0): (0, 0, 0, 1),
                  (0, 1, 0, 1): (0, 1, 0, 1),
                  (0, 1, 1, 0): (1, 0, 0, 1),
                  (0, 1, 1, 1): (1, 1, 0, 1),
                  (1, 0, 0, 0): (0, 0, 1, 0),
                  (1, 0, 0, 1): (0, 1, 1, 0),
                  (1, 0, 1, 0): (1, 0, 1, 0),
                  (1, 0, 1, 1): (1, 1, 1, 0),
                  (1, 1, 0, 0): (0, 0, 1, 1),
                  (1, 1, 0, 1): (0, 1, 1, 1),
                  (1, 1, 1, 0): (1, 0, 1, 1),
                  (1, 1, 1, 1): (1, 1, 1, 1)}

NUMBER_OF_CORES = mp.cpu_count()


class A:
    def __init__(self):
        self.triads_df = None
        self.graph = None
        self.matrix = None
        self.undirected_graph = None

    @staticmethod
    def build_triads_df(graph):
        triads_data = {str(triad): 0 for triad in TRIADS_TYPES}
        degrees_data = {'positive_in_degree(u)': [0], 'positive_out_degree(u)': [0], 'negative_in_degree(u)': [0],
                        'negative_out_degree(u)': [0], 'positive_in_degree(v)': [0], 'positive_out_degree(v)': [0],
                        'negative_in_degree(v)': [0], 'negative_out_degree(v)': [0], 'edge_sign': [0]}
        triads_data.update(degrees_data)
        triads_index = []
        nodes_with_data = set(itertools.chain(*list(graph.edges)))
        for (u, v) in itertools.combinations(nodes_with_data, 2):
            triads_index.extend([(u, v), (v, u)])

        triads_df = pd.DataFrame(triads_data, triads_index)
        return triads_df

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

    def process_frame(self, triads_df):
        for (u, v), row in triads_df.iterrows():
            if self.matrix[u, v] != 0:
                triads_df.at[(u, v), 'edge_sign'] = self.matrix[u, v]

            triads_dict_for_pair = {triad: 0 for triad in TRIADS_TYPES}
            for w in sorted(nx.common_neighbors(self.undirected_graph, u, v)):
                for triad in TRIADS_TYPES:
                    triad_status = self.get_triad_status(u, w, v, triad, self.matrix)
                    if triad_status:
                        triads_dict_for_pair[triad] += 1
            for triad in triads_dict_for_pair.keys():
                triads_df.at[(u, v), str(triad)] = triads_dict_for_pair[triad]

            count = 0
            for node in (u, v):
                node_id = int(node)
                triads_df.at[(u, v), 'positive_in_degree({})'.format('v' if count else 'u')] = self.positive_in[node_id]
                triads_df.at[(u, v), 'positive_out_degree({})'.format('v' if count else 'u')] = self.positive_out[node_id]
                triads_df.at[(u, v), 'negative_in_degree({})'.format('v' if count else 'u')] = self.negative_in[node_id]
                triads_df.at[(u, v), 'negative_out_degree({})'.format('v' if count else 'u')] = self.negative_out[node_id]

                count += 1

        return triads_df

    def compute_triads(self, tsv_file):
        graph = helper.build_graph(tsv_file)
        self.graph = graph
        self.undirected_graph = graph.to_undirected()
        triads_df = self.build_triads_df(graph)
        self.matrix = nx.to_numpy_matrix(graph, weight="weight")

        self.positive_in = np.where(self.matrix < 0, 0, self.matrix).sum(axis=0)
        self.positive_out = np.where(self.matrix < 0, 0, self.matrix).sum(axis=1)
        self.negative_in = -1*np.where(self.matrix > 0, 0, self.matrix).sum(axis=0)
        self.negative_out = -1*np.where(self.matrix > 0, 0, self.matrix).sum(axis=1)

        triads_df_split = np.array_split(triads_df, NUMBER_OF_CORES)
        pool = mp.Pool(NUMBER_OF_CORES)
        parts = pool.map(self.process_frame, triads_df_split)
        df = pd.concat(parts)
        pool.close()
        pool.join()

        return df


if __name__ == '__main__':
    print("slashdot start")

    t1 = time_of_start_computation = datetime.now()
    td = A().compute_triads("./soc-sign-Slashdot090221.tsv")
    td.to_csv("slashdot-full-features-sec", sep="\t")
    t2 = datetime.now()
    triads_time = t2 - t1
    print(triads_time)

    print("epinions start")

    t3 = time_of_start_computation = datetime.now()
    td = A().compute_triads("./soc-sign-epinions.tsv")
    td.to_csv("epinions-full-features", sep="\t")
    t4 = datetime.now()
    triads_time = t4 - t3
    print(triads_time)

