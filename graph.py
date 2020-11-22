import networkx as nx 
from IPython.display import Image

def build_graph(tsv):
    graph = nx.DiGraph();
    tsv_file = open(tsv, "r")
    lines = tsv_file.readlines()[:10]
    edges = []
    # First line contains tsv headers
    for line in lines[1:]:
        edge_details = line.strip().split('\t')
        graph.add_edge(edge_details[0], edge_details[1], weight = edge_details[2])
        #graph.edges[edge_details[0], edge_details[1]]['weight'] = edge_details[2]
        #print(edge_details[0], edge_details[1], edge_details[2])
    
    graph.add_weighted_edges_from(edges)
    
        
    return graph


if __name__ == "__main__":
    graph = build_graph("./datasets/soc-sign-epinions.tsv")
    print(graph.edges())
