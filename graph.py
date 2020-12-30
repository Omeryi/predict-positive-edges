import networkx as nx
import matplotlib.pyplot as plt
import csv
import pandas as pd
import random
from datetime import datetime
import scipy.sparse as sp
import numpy as np

TSV_FIELDS = ["FromNodeId", "ToNodeId", "Sign"]


# TODO CHANGE TO WORK WITH PANDAS
def build_graph(tsv, record_fake_data=False):
    graph = nx.DiGraph()
    tsv_file = open(tsv, "r")
    df = pd.read_csv(tsv, sep="\t")
    # skip headers line:
    next(tsv_file)

    real_nodes = set()

    for i in range(df.max().max()):
        if i not in graph.nodes():
            graph.add_node(i)
    for line in csv.reader(tsv_file, delimiter="\t"):
        graph.add_edge(int(line[0]), int(line[1]), weight=int(line[2]))
        real_nodes.add(int(line[0]))
        real_nodes.add(int(line[1]))

    if record_fake_data:
        return graph, real_nodes
    else:
        return graph


def graph_to_tsv(graph, output_file_name):
    with open(output_file_name, "w", newline="") as tsvfile:
        fieldnames = TSV_FIELDS
        writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=fieldnames)

        writer.writeheader()

        for (u, v) in graph.edges():
            writer.writerow({TSV_FIELDS[0]: u, TSV_FIELDS[1]: v, TSV_FIELDS[2]: graph[u][v]["weight"]})


def create_graph_variants(tsv, name):
    type1graph = build_graph(tsv)
    type2graph = type1graph.copy()
    type3graph = type1graph.copy()
    type4graph = type1graph.copy()
    print("HI")
    negative_edges = []
    positive_edges = []
    num_negative_edges = 0
    random_positive_edges = []

    for (u, v) in type1graph.edges():
        if type1graph[u][v]["weight"] == -1:
            negative_edges.append((u, v))
            num_negative_edges = num_negative_edges + 1

    type2graph.remove_edges_from(negative_edges)
    positive_edges = type2graph.edges()
    random_positive_edges = random.sample(positive_edges, num_negative_edges)

    type3graph.remove_edges_from(negative_edges)
    type3graph.remove_edges_from(random_positive_edges)

    type4graph.remove_edges_from(random_positive_edges)

    graph_to_tsv(type2graph, "variant2-" + name)
    graph_to_tsv(type3graph, "variant3-" + name)
    graph_to_tsv(type4graph, "variant4-" + name)


def draw_graph(graph):
    pos = nx.spring_layout(graph)
    plt.figure(30, figsize=(30, 30))
    nx.draw(graph, pos, node_size=1500, font_size=10, with_labels=True)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'weight'))


if __name__ == "__main__":
    t1 = time_of_start_computation = datetime.now()

    print("wiki:")
    create_graph_variants("./datasets/wiki.tsv", "wiki")
    print("slashdot:")
    create_graph_variants("./datasets/soc-sign-Slashdot090221.tsv", "slashdot")
    print("epinions:")
    create_graph_variants("./datasets/soc-sign-epinions.tsv", "epinions")

    t2 = datetime.now()
    triads_time = t2 - t1
    print(triads_time)
