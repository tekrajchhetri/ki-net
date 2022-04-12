# -*- coding: utf-8 -*-
# @Time    : 18.03.22 14:46
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : https://tekrajchhetri.com/
# @File    : graph_learn.py
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
import types
from owlready2 import *

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


    print(f"structure learning initiated with tabuedge:{tabuedge}, "
          f"use_bias:{use_bias},"
          f"use_gpu:{use_gpu}, "
          f"max_iter:{max_iter},"
          f"hidden_layer_units={hidden_layer_units},"
          f"lasso_beta:{lasso_beta}, "
          f"domainknowledge={domainknowledge} "
          f"ridge_beta:{ridge_beta}")
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
        sm.add_edges_from(domainknowledge,
                                   origin="expert",
                                   weight=1)
    dag_graph = nx.MultiDiGraph(sm)
    if is_dag(dag_graph):
        return nx.MultiDiGraph(sm)
    else:
        return None

def capitalise_word(input_word):
    cap_word =  ''.join([word.capitalize() for word in input_word.split()])
    return cap_word.strip()


def transform_graph_to_ontology(G):
    """Take input networkx MultiDiGraph and convert to ontology
    :param G: networkx MultiDiGraph
    :return: owl ontology or dictionary
    """
    class_mapper = {}
    object_class_mapper = {}
    if type(G) == nx.classes.multidigraph.MultiDiGraph:
        onto_sensor = get_ontology("https://finaltestobj.owl#")
        with onto_sensor:
            for class_name in list(G.nodes()):
                capitalise_class_name = capitalise_word(class_name)
                cclass_name = types.new_class(f"{capitalise_class_name}", (Thing,))
                class_mapper[capitalise_class_name] = cclass_name

            for g in list(G.adjacency()):
                if bool(g[1]):
                    objpropclass_name = types.new_class(f"is_influenced_by_{capitalise_word(g[0])}", (ObjectProperty,))
                    object_class_mapper[f"is_influenced_by_{capitalise_word(g[0])}"] = objpropclass_name

                    datapropclass_name = types.new_class("has_influence_factor_of", (DataProperty, FunctionalProperty,))
                    class_mapper[capitalise_word(g[0])].is_a.append(datapropclass_name.some(float))

                    for restriction_class_name in [capitalise_word(word) for word in list(g[1].keys())]:
                        datapropclass_name_origin = types.new_class("origin_from", (DataProperty, FunctionalProperty,))
                        class_mapper[restriction_class_name].is_a.append(datapropclass_name_origin.some(str))
                        class_mapper[capitalise_word(g[0])].is_a.append(
                            object_class_mapper[f"is_influenced_by_{capitalise_word(g[0])}"].some(
                                class_mapper[restriction_class_name]))
        return onto_sensor


    else:
        return {"message": "Graph must be of type nx.classes.multidigraph.MultiDiGraph", "status": 0}


def to_convert_to_SDOW_format(G):
    """ Convert to dataframe for KG
    :param graph: networkx graph
    :return: pandas DataFrame
    """
    list_for_df = []
    graph = G.adjacency()
    if is_networkxgraph(G):
        for adjacency_graph_dictionary in graph:
            # print(adjacency_graph_dictionary)
            source = adjacency_graph_dictionary[0]
            for dest in adjacency_graph_dictionary[1]:
                if type(G) == nx.classes.multidigraph.MultiDiGraph:
                    make_df_list = [source,
                                        dest,
                                        adjacency_graph_dictionary[1][dest][0]["origin"],
                                        adjacency_graph_dictionary[1][dest][0]["weight"]
                                        ]
                    list_for_df.append(make_df_list)

                elif type(G) == nx.classes.multidigraph.DiGraph:
                    make_df_list = [source,
                                        dest,
                                        adjacency_graph_dictionary[1][dest]["origin"],
                                        adjacency_graph_dictionary[1][dest]["weight"]
                                        ]
                    list_for_df.append(make_df_list)



        return pd.DataFrame(list_for_df, columns=["Source", "Destination", "Origin", "Weight"])

def is_same(source, destination):
    if source == destination:
        return True
    else:
        return False

def same_node_error():
    return {"message": "Invalid selection, source and destination cannot contain same node.", "status": 0}


def no_destination_selected():
    return {"message": "Destination cannot be empty.", "status": 0}



def make_edges(source, destination):
    """Transform selected node options to graphs
    :param source: Selected source node to make connection with destination node
    :param destination: Destination node to be connected to source node
    :return: dictionary
    """
    make_edges = []
    k = len(destination) - 1
    if len(source) < len(destination):
        return {"message": invalid_selection(), "status":0}
    else:
        if len(destination) > 0:
            for i in range(len(source)):
                if i > k:
                    if not is_same(source[i], destination[k]):
                        make_edges.append((source[i], destination[k]))
                    else:
                        return same_node_error()

                else:
                    if not is_same(source[i], destination[k]):
                        make_edges.append((source[i], destination[i]))
                    else:
                        return same_node_error()
        else:
            return no_destination_selected()

    if len(make_edges) <=0:
        return {"message":"No selection of Edges made", "status": 0}
    else:
        return  {"message": make_edges, "status":1}




def display_learned_graph(graph):
    """convert to agraph for displaying
    :param graph: networkx graph
    :return: agraph
    """

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
    """initiate the structure learning process
    :param dataset: input dataset i.e., dataframe
    :param threshold: filter threshold
    :param domainknowledge: domain knowledge or the connection between nodes that should exist
    :param tabuedge: connection that is forbidden
    :param use_bias:  Whether to fit a bias parameter in the NOTEARS algorithm
    :param use_gpu: If take benefits of available GPU for learning
    :param max_iter: steps during optimisation
    :param hidden_layer_units: hidden layers for mlp
    :param lasso_beta: l1 regularization
    :param ridge_beta: l2 regularization
    :return: learned networkx graph or boolean
    """
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
            return graph
        else:
            return False
    except  Exception as e:
        print(e)


