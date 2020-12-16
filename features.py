import pandas as pd
import graph as helper
import networkx as nx
import itertools
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from datetime import datetime


def compute_degrees(tsv_file):
    # build new data frame for degrees:
    df = pd.read_csv(tsv_file, sep='\t')
    graph = helper.build_graph(tsv_file)
    data = {'positive_in_degree': [0], 'positive_out_degree': [0], 'negative_in_degree': [0],
            'negative_out_degree': [0]}
    degrees_df = pd.DataFrame(data, index=[int(node) for node in graph.nodes()])

    # fill new data frame with data:
    for node in graph.nodes():
        node_id = int(node)
        positive_in = df[(df.ToNodeId == node_id) & (df.Sign == 1)].shape[0]
        positive_out = df[(df.FromNodeId == node_id) & (df.Sign == 1)].shape[0]
        negative_in = df[(df.ToNodeId == node_id) & (df.Sign == -1)].shape[0]
        negative_out = df[(df.FromNodeId == node_id) & (df.Sign == -1)].shape[0]

        degrees_df.at[node, 'positive_in_degree'] = positive_in
        degrees_df.at[node, 'positive_out_degree'] = positive_out
        degrees_df.at[node, 'negative_in_degree'] = negative_in
        degrees_df.at[node, 'negative_out_degree'] = negative_out

    return degrees_df


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
        self.graph_df = None
        self.graph = None
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
    def get_triad_status(u, w, v, triad, graph_df):
        first_edge = 0
        second_edge = 0

        sign1 = -1 if triad[1] else 1
        sign2 = -1 if triad[3] else 1

        if (not triad[0]):
            first_edge = \
                graph_df[
                    (graph_df.ToNodeId == int(w)) & (graph_df.FromNodeId == int(u)) & (graph_df.Sign == sign1)].shape[0]
        else:
            first_edge = \
                graph_df[
                    (graph_df.ToNodeId == int(u)) & (graph_df.FromNodeId == int(w)) & (graph_df.Sign == sign1)].shape[0]

        if (not triad[2]):
            second_edge = \
                graph_df[
                    (graph_df.ToNodeId == int(w)) & (graph_df.FromNodeId == int(v)) & (graph_df.Sign == sign2)].shape[0]
        else:
            second_edge = \
                graph_df[
                    (graph_df.ToNodeId == int(v)) & (graph_df.FromNodeId == int(w)) & (graph_df.Sign == sign2)].shape[0]

        return 1 if (first_edge and second_edge) else 0

    def process_frame(self, triads_df):
        graph_df = self.graph_df
        for (u, v), row in triads_df.iterrows():
            if (graph_df[(graph_df.ToNodeId == int(v)) & (graph_df.FromNodeId == int(u))].any()[0]):
                triads_df.at[(u, v), 'edge_sign'] = graph_df[(graph_df.ToNodeId == int(v)) & (graph_df.FromNodeId == int(u))].Sign


            triads_dict_for_pair = {triad: 0 for triad in TRIADS_TYPES}
            for w in sorted(nx.common_neighbors(self.undirected_graph, u, v)):
                for triad in TRIADS_TYPES:
                    triad_status = self.get_triad_status(u, w, v, triad, self.graph_df)
                    if triad_status:
                        triads_dict_for_pair[triad] += 1
            for triad in triads_dict_for_pair.keys():
                triads_df.at[(u, v), str(triad)] = triads_dict_for_pair[triad]

            count = 0
            for node in (u, v):
                node_id = int(node)
                positive_in = graph_df[(graph_df.ToNodeId == node_id) & (graph_df.Sign == 1)].shape[0]
                positive_out = graph_df[(graph_df.FromNodeId == node_id) & (graph_df.Sign == 1)].shape[0]
                negative_in = graph_df[(graph_df.ToNodeId == node_id) & (graph_df.Sign == -1)].shape[0]
                negative_out = graph_df[(graph_df.FromNodeId == node_id) & (graph_df.Sign == -1)].shape[0]

                triads_df.at[(u, v), 'positive_in_degree({})'.format('v' if count else 'u')] = positive_in
                triads_df.at[(u, v), 'positive_out_degree({})'.format('v' if count else 'u')] = positive_out
                triads_df.at[(u, v), 'negative_in_degree({})'.format('v' if count else 'u')] = negative_in
                triads_df.at[(u, v), 'negative_out_degree({})'.format('v' if count else 'u')] = negative_out

                count +=1

        return triads_df

    def compute_triads(self, tsv_file):
        self.graph_df = pd.read_csv(tsv_file, sep='\t')
        graph = helper.build_graph(tsv_file)
        self.graph = graph
        self.undirected_graph = graph.to_undirected()
        triads_df = self.build_triads_df(graph)

        triads_df_split = np.array_split(triads_df, NUMBER_OF_CORES)
        pool = mp.Pool(NUMBER_OF_CORES)
        parts = pool.map(self.process_frame, triads_df_split)
        df = pd.concat(parts)
        pool.close()
        pool.join()
        return df


if __name__ == '__main__':
    time_of_start_computation = datetime.now()
    td = A().compute_triads("./datasets/wiki-demo-10.tsv")
    td.to_csv("wiki-features-10-lines-v1", sep="\t")
    time_of_end_computation = datetime.now()
    triads_time = time_of_end_computation - time_of_start_computation
    print(triads_time)


