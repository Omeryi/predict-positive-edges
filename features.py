import pandas as pd
import graph as helper
import networkx as nx
import itertools
import numpy as np
import multiprocessing as mp
import logging

""" Prediction of positive edge is based on common triads for both nodes. 
    An embeddedbess treshold is set to ensure the prediction and train will be done based on node couples which has
    enough triads data
"""
EMBEDDEDNESS_TRESHOLD = 25
FEATURES_PATH = "./calculated_features/"

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

"""Each triad type is represented by four digits as each digit may be 0 or 1.
   Let w be the common neighbor of two nodes in a triad, u and v:
   The first digit represents the direction of the edge between u, w: 0 if u points to w, else 1.
   The second digit represents the sign of the edge between u, w: 0 if it`s positive, else 1.
   The third digit represents the direction of the edge between v, w: 0 if v points to w, else 1.
   The fourth digit represents the sign of the edge between v, w: 0 if it`s positive, else 1.
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

# Parallel computing of the features helps to reduce the calculating runtime
NUMBER_OF_CORES = mp.cpu_count()


class Features_calculator:
    def __init__(self):
        self.features_df = None
        self.graph = None
        self.matrix = None
        self.undirected_graph = None
        self.embeddedness_matrix = None


    def build_features_df(self, graph):
        triads_data = {str(triad): 0 for triad in TRIADS_TYPES}
        additional_data = {'edge_sign': [0]}

        triads_data.update(additional_data)
        triads_index = []
        nodes_with_data = set(itertools.chain(*list(graph.edges)))
        for (u, v) in itertools.combinations(nodes_with_data, 2):
            # include in the features only nodes that pass the embeddedbess treshold
            if self.embeddedness_matrix[int(u), int(v)] >= EMBEDDEDNESS_TRESHOLD:
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
            # Fill the edge sign:
            if self.matrix[u, v] != 0:
                features_df.at[(u, v), 'edge_sign'] = self.matrix[u, v]

            # Count the common triads for each pair:
            triads_dict_for_pair = {triad: 0 for triad in TRIADS_TYPES}
            for w in sorted(nx.common_neighbors(self.undirected_graph, u, v)):
                for triad in TRIADS_TYPES:
                    triad_status = self.get_triad_status(u, w, v, triad, self.matrix)
                    if triad_status:
                        triads_dict_for_pair[triad] += 1

            for triad in triads_dict_for_pair.keys():
                features_df.at[(u, v), str(triad)] = triads_dict_for_pair[triad]

        return features_df


    def compute_features(self, tsv_file):
        graph = helper.build_graph(tsv_file)
        self.graph = graph
        self.undirected_graph = graph.to_undirected()
        self.embeddedness_matrix = helper.get_embeddedbess_matrix(graph)
        features_df = self.build_features_df(graph)
        self.matrix = nx.to_numpy_matrix(graph, weight="weight")

        # Parallel calculation of the features, each thread calculates a different chunk
        features_df_split = np.array_split(features_df, NUMBER_OF_CORES)
        pool = mp.Pool(NUMBER_OF_CORES)
        parts = pool.map(self.process_frame, features_df_split)
        df = pd.concat(parts)
        pool.close()
        pool.join()
        return df


if __name__ == '__main__':
    logging.info("start")

    td = Features_calculator().compute_features("./datasets/wiki-demo-1000.tsv")
    td.to_csv(FEATURES_PATH + "assertion-1000.tsv", sep="\t")

    logging.info("V1")
    td = Features_calculator().compute_features("./datasets/variant1-wiki.tsv")
    td.to_csv(FEATURES_PATH +"var1-features.tsv", sep="\t")

    logging.info("V2")
    td = Features_calculator().compute_features("./datasets/variant2-wiki.tsv")
    td.to_csv(FEATURES_PATH + "var2-features.tsv", sep="\t")

    logging.info("V3")
    td = Features_calculator().compute_features("./datasets/variant3-wiki.tsv")
    td.to_csv(FEATURES_PATH + "var3-features.tsv", sep="\t")

    logging.info("V4")
    td = Features_calculator().compute_features("./datasets/variant4-wiki.tsv")
    td.to_csv(FEATURES_PATH + "var4-features.tsv", sep="\t")

