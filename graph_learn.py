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
from helper import get_owl_file
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
    cap_word = ''.join([word.capitalize() for word in input_word.split()])
    return cap_word.strip()


def annotate_ontology(G, ontology, class_mapper, object_class_mapper, originated_domain, influence_domain):
    """ Annotate the ontology
    :param G: networkx MultiDiGraph
    :param ontology:  owlready2 instance of ontology or schema
    :param class_mapper: owlready2 created ontology classes
    :param object_class_mapper:  object properties
    :param originated_domain:  data property
    :param influence_domain: data property
    :return: annotated ontology
    """
    annotated_ontology = ontology
    with annotated_ontology:
        for g in list(G.adjacency()):
            if bool(g[1]):
                for k in list(g[1].keys()):
                    dataclass = class_mapper[capitalise_word(k)](capitalise_word(k).lower())
                    object_class_mapper[f"isInfluencedBy{capitalise_word(g[0])}"][dataclass].append(
                        class_mapper[capitalise_word(g[0])])
                    originated_domain[f"isOriginatedFrom{capitalise_word(g[0])}"][dataclass].append(
                        g[1][k][0]["origin"])
                    influence_domain[f"hasInfluenceFactorOf{capitalise_word(g[0])}"][dataclass].append(
                        g[1][k][0]["weight"])
    return annotated_ontology



def transform_graph_to_ontology(G):
    """ Transform graph G into Ontology and annotate it
    :param G: Networkx Graph
    :return: dict indicating success or failure
    """
    class_mapper = {}
    object_class_mapper = {}
    originated_domain = {}
    influence_domain = {}
    if type(G) == nx.classes.multidigraph.MultiDiGraph:
        onto_sensor = get_ontology("https://checking.owl#")
        with onto_sensor:
            for class_name in list(G.nodes()):
                capitalise_class_name = capitalise_word(class_name)
                cclass_name = types.new_class(f"{capitalise_class_name}", (Thing,))
                class_mapper[capitalise_class_name] = cclass_name

            for g in list(G.adjacency()):
                if bool(g[1]):
                    objpropclass_name = types.new_class(f"isInfluencedBy{capitalise_word(g[0])}", (ObjectProperty,))
                    objpropclass_name.domain.append(class_mapper[capitalise_word(g[0])])
                    objpropclass_name.range = [class_mapper[capitalise_word(word)] for word in list(g[1].keys())]
                    object_class_mapper[f"isInfluencedBy{capitalise_word(g[0])}"] = objpropclass_name

                for restriction_class_name in [word for word in list(g[1].keys())]:
                    datapropclass_name_origin = types.new_class(f"isOriginatedFrom{capitalise_word(g[0])}",
                                                                (DataProperty, FunctionalProperty,))
                    datapropclass_name_origin.domain.append(class_mapper[capitalise_word(restriction_class_name)])
                    datapropclass_name_origin.range.append(str)
                    originated_domain[f"isOriginatedFrom{capitalise_word(g[0])}"] = datapropclass_name_origin

                    datapropclass_name = types.new_class(f"hasInfluenceFactorOf{capitalise_word(g[0])}",
                                                         (DataProperty, FunctionalProperty,))
                    datapropclass_name.domain.append(class_mapper[capitalise_word(restriction_class_name)])
                    datapropclass_name.range = [float]
                    influence_domain[f"hasInfluenceFactorOf{capitalise_word(g[0])}"] = datapropclass_name

        onto_save = annotate_ontology(G, onto_sensor, class_mapper, object_class_mapper, originated_domain,
                                      influence_domain)
        onto_save.save(get_owl_file())
        return {"status": 1, "message": "success"}

    else:
        return {"status": 0, "message": "Graph must be of type nx.classes.multidigraph.MultiDiGraph"}

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


