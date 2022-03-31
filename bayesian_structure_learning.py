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



def start_linear_structure_learning(dataset, threshold, domainknowledge=None,
                                    tabuedge=None,
                                    use_bias=True,
                                    use_gpu=False,
                                    max_iter=100,
                                    hidden_layer_units=None,
                                    lasso_beta=0.1,
                                    ridge_beta=0.2):
    """Perform the Bayesian structure learning
    :param dataset: pandas dataframe
    :param threshold: threshold for graph filtering
    :param tabuedge: edges to be removed
    :param use_bias: Whether to fit a bias parameter in the NOTEARS algorithm
    :param use_gpu: GPU to use?
    :param max_iter: steps during optimisation
    :param hidden_layer_units: hidden layer units
    :param lasso_beta: l1 regularisation
    :param ridge_beta: l2 regularisation
    :return:learned structure as networkx graph
    """


    print(f"structure learning initiated with tabuedge:{tabuedge}, use_bias:{use_bias},use_gpu:{use_gpu}, max_iter:{max_iter},hidden_layer_units={hidden_layer_units},"
          f"lasso_beta:{lasso_beta},ridge_beta:{ridge_beta}")
    sm = from_pandas(dataset,
                         tabu_edges=tabuedge,
                         use_bias=use_bias,
                         use_gpu=use_gpu,
                         max_iter=max_iter,
                         hidden_layer_units=hidden_layer_units,
                         lasso_beta=lasso_beta,
                         ridge_beta=ridge_beta
                         )
    sm.remove_edges_below_threshold(threshold)
    if domainknowledge is not None:
        sm.add_edges_from(domainknowledge, origin="expert")
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

def make_edges(source, destination,isdomain=False):
    make_edges = []
    message = {}
    if len(source) != len(destination):
        return {"message": invalid_selection(), "status":0}
    else:
        for i in range(len(source)):
            if source[i] == destination[i]:
                return {"message": "Invalid selection, source and destination cannot contain same node.", "status": 0}
            else:
                make_edges.append((source[i], destination[i]))
        if len(make_edges) <=0:
            return {"message":"No selection of Edges made", "status": 0}
        else:
            return  {"message": make_edges, "status":1}




def display_learned_graph(graph):

    return_value = agraph(nodes=convert_agraph_node(graph),
                          edges=convert_agraph_edge(graph),
                          config=gaph_config())
    return return_value

def init_learning_process(dataset, threshold, domainknowledge,
                                    tabuedge,
                                    use_bias,
                                    use_gpu,
                                    max_iter,
                                    hidden_layer_units,
                                    lasso_beta,
                                    ridge_beta):
    print("learning init")
    graph = start_linear_structure_learning(dataset=dataset,
                                            domainknowledge=domainknowledge,
                                            threshold=threshold,
                                            use_bias=use_bias,
                                            use_gpu=use_gpu,
                                            max_iter=max_iter,
                                            hidden_layer_units=hidden_layer_units,
                                            lasso_beta=lasso_beta,
                                            ridge_beta=ridge_beta,
                                            tabuedge=tabuedge)
    try:
        if graph is not None:
            return display_learned_graph(graph)
        else:
            return zero_error()
    except  Exception as e:
        print(e)


