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
from helper import zero_error, is_networkxgraph

def is_dag(G):
    """Checks if the graph is DAG
    :param nx_graph: networkx graph
    :return: boolean
    """
    return nx.is_directed_acyclic_graph(G)



def start_linear_structure_learning(dataset, threshold):
    """Perform the Bayesian structure learning
    :param dataset: pandas dataframe
    :return: learned structure
    """
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

def display_learned_graph(graph):

    return_value = agraph(nodes=convert_agraph_node(graph),
                          edges=convert_agraph_edge(graph),
                          config=gaph_config())
    return return_value

def init_learning_process(datasets, threshold):
    graph = start_linear_structure_learning(dataset=datasets, threshold=threshold)
    try:
        if graph is not None:
            return display_learned_graph(graph)
        else:
            return zero_error()
    except  Exception as e:
        print(e)


