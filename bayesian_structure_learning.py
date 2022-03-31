# -*- coding: utf-8 -*-
# @Time    : 18.03.22 14:46
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : https://tekrajchhetri.com/
# @File    : bayesian_structure_learning.py
# @Software: PyCharm

from causalnex.structure.pytorch.notears import from_pandas
import networkx as nx
import pandas as pd
from streamlit_agraph import agraph
from helper import convert_agraph_node
from helper import convert_agraph_edge
from helper import gaph_config
from helper import zero_error
from helper import  is_networkxgraph
from helper import invalid_selection

def is_dag(G):
    """Checks if the graph is DAG
    :param nx_graph: networkx graph
    :return: boolean
    """
    return nx.is_directed_acyclic_graph(G)



def start_linear_structure_learning(dataset, threshold, tabuedge=None):
    """Perform the Bayesian structure learning
    :param dataset: pandas dataframe
    :return: learned structure
    """
    if tabuedge is not None:
        sm = from_pandas(dataset, tabu_edges=tabuedge)
    else:
        sm = from_pandas(dataset)
    sm.remove_edges_below_threshold(threshold)
    dag_graph = nx.MultiDiGraph(sm)
    if is_dag(dag_graph):
        return nx.MultiDiGraph(sm)
    else:
        return None


def to_convert_to_SDOW_format(G):
    """ Convert to dataframe for KG
    :param graph: networkx graph
    :return: pandas DataFrame
    """
    graph = G.adjacency()
    if is_networkxgraph(G):
        list_for_df = []
        for adjacency_graph_dictionary in graph:
            source = adjacency_graph_dictionary[0]
            for dest in adjacency_graph_dictionary[1]:
                if type(graph) == nx.classes.multidigraph.MultiDiGraph:
                    list_for_df.append([source,
                                        dest,
                                        adjacency_graph_dictionary[1][dest][0]["origin"],
                                        adjacency_graph_dictionary[1][dest][0]["weight"]
                                        ])

                elif type(graph) == nx.classes.multidigraph.DiGraph:
                    list_for_df.append([source,
                                        dest,
                                        adjacency_graph_dictionary[1][dest]["origin"],
                                        adjacency_graph_dictionary[1][dest]["weight"]
                                        ])

        return pd.DataFrame(list_for_df)

def make_tabu_edges(source, destination):
    tabu_edges = []
    message = {}
    if len(source) != len(destination):
        return {"message": invalid_selection(), "status":0}
    else:
        for i in range(len(source)):
            if source[i] == destination[i]:
                return {"message": "Invalid selection, source and destination cannot contain same node.", "status": 0}
            else:
                tabu_edges.append((source[i], destination[i]))
        if len(tabu_edges) <=0:
            return {"message":"No selection of Tabu Edges made", "status": 0}
        else:
            return  {"message": tabu_edges, "status":1}




def display_learned_graph(graph):

    return_value = agraph(nodes=convert_agraph_node(graph),
                          edges=convert_agraph_edge(graph),
                          config=gaph_config())
    return return_value

def init_learning_process(datasets, threshold, tabuedge):
    graph = start_linear_structure_learning(dataset=datasets, threshold=threshold, tabuedge=tabuedge)
    try:
        if graph is not None:
            return display_learned_graph(graph)
        else:
            return zero_error()
    except  Exception as e:
        print(e)


