import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import random
from scipy import sparse

TSV_FIELDS = ["FromNodeId", "ToNodeId", "Sign"]
DATASET_PATH = "./datasets/"

""" Builds a directed graph based on the information in the tsv file given.
    As the basis of runtime optimizations, the graph will be represented as adjacency matrix A in which
    cell A[i][j] = edge sign between i to j. Thus, nodes which are not existing in the graph were added to it to fill 
    the spaces between existing nodes. 
    The wikipedia graph contains about 1000 non-real nodes and 7000 real nodes.
    The existing nodes are being recorded for correct future use (features building based on real nodes, selecting non
    edges from the graph for 10-fold varient). 
"""


def build_graph(tsv, record_fake_data=False):
    graph = nx.DiGraph()
    df = pd.read_csv(tsv, sep="\t")
    real_nodes = set()

    # Add all numbers from 0 to the highest node id to the graph
    for i in range(df.max().max()):
        if i not in graph.nodes():
            graph.add_node(i)

    # Record the real nodes
    for _, line in df.iterrows():
        graph.add_edge(int(line.FromNodeId), int(line.ToNodeId), weight=int(line.Sign))
        real_nodes.add(int(line.FromNodeId))
        real_nodes.add(int(line.ToNodeId))

    if record_fake_data:
        return graph, real_nodes
    else:
        return graph


# This function is used to create the tsv files of 3 additional graph variants we create
def graph_to_tsv(graph, output_file_name):
    with open(DATASET_PATH + output_file_name + ".tsv", "w", newline="") as tsvfile:
        fieldnames = TSV_FIELDS
        writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=fieldnames)

        writer.writeheader()

        for (u, v) in graph.edges():
            writer.writerow({TSV_FIELDS[0]: u, TSV_FIELDS[1]: v, TSV_FIELDS[2]: graph[u][v]["weight"]})


""" There are 4 graph variants:
    Variant 1: the original graph
    Variant 2: the graph without all negative edges N 
    Variant 3: the graph without all negative edges and without a group of randomly selected positive edges P+ such
               that |N| = |P+|
    Variant 4: the graph without all edges from the group P+
    
    All variants are being written to tsv files for future computation of features 
"""


def create_graph_variants(tsv, name):
    type1graph = build_graph(tsv)
    type2graph = type1graph.copy()
    type3graph = type1graph.copy()
    type4graph = type1graph.copy()

    negative_edges = []
    positive_edges = []
    num_negative_edges = 0
    random_positive_edges = []

    for (u, v) in type1graph.edges():
        if type1graph[u][v]["weight"] == -1:
            negative_edges.append((u, v))
            num_negative_edges = num_negative_edges + 1

    # Remove the negative edges to create variant2:
    type2graph.remove_edges_from(negative_edges)
    positive_edges = type2graph.edges()
    random_positive_edges = random.sample(positive_edges, num_negative_edges)

    # Remove the negative edges and edges from P+ to create variant2:
    type3graph.remove_edges_from(negative_edges)
    type3graph.remove_edges_from(random_positive_edges)

    # Remove the edges from P+ to create variant2:
    type4graph.remove_edges_from(random_positive_edges)

    # Write the 3 additional variants to tsv files
    graph_to_tsv(type2graph, "variant2-" + name)
    graph_to_tsv(type3graph, "variant3-" + name)
    graph_to_tsv(type4graph, "variant4-" + name)


# This one was just for fun
def draw_graph(graph):
    pos = nx.spring_layout(graph)
    plt.figure(10, figsize=(10, 10))
    nx.draw(graph, pos, node_size=1500, font_size=10, with_labels=True)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'weight'))


def get_embeddedbess_matrix(graph):
    A = sparse.csr_matrix(nx.to_numpy_matrix(graph, weight="weight"))
    adj = A.todense()
    A[A != 0] = 1
    B = A + A.T
    embeddedness_matrix = (B @ B.T).todense()
    return embeddedness_matrix


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

if __name__ == "__main__":
    print("Creating 3 additional graph variants:")
    create_graph_variants("./datasets/variant1-wiki.tsv", "wiki")



