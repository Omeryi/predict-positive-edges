import networkx as nx 
import matplotlib.pyplot as plt
import csv
import pandas as pd

# TODO CHANGE TO WORK WITH PANDAS
def build_graph(tsv):
    graph = nx.DiGraph()
    tsv_file = open(tsv, "r")
    df = pd.read_csv(tsv, sep="\t")
    # skip headers line:
    next(tsv_file)
    # TODO FIND ACTUAL MAX
    for i in range(df.max().max()):
        if i not in graph.nodes():
            graph.add_node(i)
    for line in csv.reader(tsv_file, delimiter="\t"):
        graph.add_edge(int(line[0]), int(line[1]), weight = int(line[2]))
    return graph


def draw_graph(graph):
    pos = nx.spring_layout(graph)   
    plt.figure(30,figsize=(30,30)) 
    nx.draw(graph, pos, node_size = 1500, font_size = 10 , with_labels = True)   
    nx.draw_networkx_edge_labels(graph,pos, edge_labels=nx.get_edge_attributes(graph,'weight'))


if __name__ == "__main__":
    graph = build_graph("./datasets/soc-sign-epinions.tsv")
    draw_graph(graph)
