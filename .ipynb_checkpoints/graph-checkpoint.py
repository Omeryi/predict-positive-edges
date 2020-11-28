import networkx as nx 
import matplotlib.pyplot as plt


# TODO CHANGE TO WORK WITH PANDAS
def build_graph(tsv):
    graph = nx.DiGraph()
    tsv_file = open(tsv, "r")
    # CHANGE : NOT 10
    lines = tsv_file.readlines()[:10]
    # First line contains tsv headers
    for line in lines[1:]:
        edge_details = line.strip().split('\t')
        graph.add_edge(edge_details[0], edge_details[1], weight = edge_details[2])
        
    return graph


def draw_graph(graph):
    pos = nx.spring_layout(graph)   
    plt.figure(30,figsize=(30,30)) 
    nx.draw(graph, pos, node_size = 1500, font_size = 10 , with_labels = True)   
    nx.draw_networkx_edge_labels(graph,pos, edge_labels=nx.get_edge_attributes(graph,'weight'))


if __name__ == "__main__":
    graph = build_graph("./datasets/soc-sign-epinions.tsv")
    draw_graph(graph)
