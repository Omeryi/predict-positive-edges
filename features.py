import pandas as pd
import graph as helper
import networkx as nx
import itertools
from datetime import datetime

def compute_degrees(tsv_file):
    # build new data frame for degrees: 
    df = pd.read_csv(tsv_file, sep='\t')
    graph = helper.build_graph(tsv_file)
    data = {'positive_in_degree': [0], 'positive_out_degree':[0], 'negative_in_degree': [0], 'negative_out_degree': [0]}
    degrees_df = pd.DataFrame(data, index = [node for node in graph.nodes()])
    
    # fill new data frame with data:
    for node in graph.nodes():
        node_id = int(node)
        positive_in = df[(df.ToNodeId == node_id) & (df.Sign == 1)].shape[0]
        positive_out = df[(df.FromNodeId == node_id) & (df.Sign == 1)].shape[0]
        negative_in = df[(df.ToNodeId == node_id) & (df.Sign == -1)].shape[0]
        negative_out = df[(df.FromNodeId == node_id) & (df.Sign == -1)].shape[0]
        
        degrees_df.at[node,'positive_in_degree'] = positive_in
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

TRIADS_TYPES = [(0,0,0,0),
               (0,0,0,1),
               (0,0,1,0), 
               (0,0,1,1),
               (0,1,0,0),
               (0,1,0,1),
               (0,1,1,0),
               (0,1,1,1),
               (1,0,0,0),
               (1,0,0,1),
               (1,0,1,0),
               (1,0,1,1),
               (1,1,0,0),
               (1,1,0,1),
               (1,1,1,0),
               (1,1,1,1)]


def get_triad_status(u, w, v, triad, graph_df):
    first_edge = 0
    second_edge = 0
    
    if (not triad[0]):
        first_edge = graph_df[(graph_df.ToNodeId == w) & (graph_df.FromNodeId == u) & (graph_df.Sign == -1 if triad[1] else 1)].shape[0]
    else:
        first_edge = graph_df[(graph_df.ToNodeId == u) & (graph_df.FromNodeId == w) & (graph_df.Sign == -1 if triad[1] else 1)].shape[0]
        
    if (not triad[2]):
        second_edge = graph_df[(graph_df.ToNodeId == w) & (graph_df.FromNodeId == v) & (graph_df.Sign == -1 if triad[3] else 1)].shape[0]
    else:
        second_edge = graph_df[(graph_df.ToNodeId == v) & (graph_df.FromNodeId == w) & (graph_df.Sign == -1 if triad[3] else 1)].shape[0]
        
    return 1 if (first_edge and second_edge) else 0
          
def compute_triads(tsv_file):
    graph_df = pd.read_csv(tsv_file, sep='\t')
    graph = helper.build_graph(tsv_file)
    undirected_graph = graph.to_undirected()
    
    triads_data = {str(triad) : [0] for triad in TRIADS_TYPES}
    triads_index = [(u,v) for (u,v) in itertools.permutations(graph.nodes(), 2)]
    triads_df = pd.DataFrame(triads_data, triads_index)
    
    for (u,v) in itertools.permutations(graph.nodes(), 2):
        for w in sorted(nx.common_neighbors(undirected_graph, u, v)):
            for triad in TRIADS_TYPES:
                triad_status = get_triad_status(u, w, v, triad, graph_df)
                print(triad_status)
                triads_df.at[(u,v), str(triad)] = triads_df.at[(u,v), str(triad)] + triad_status

    return triads_df