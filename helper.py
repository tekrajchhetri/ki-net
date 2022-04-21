# -*- coding: utf-8 -*-
# @Time    : 18.03.22 14:40
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : https://tekrajchhetri.com/
# @File    : helper.py
# @Software: PyCharm

import os
import glob
import pandas as pd
import streamlit as st
from rdflib import Graph
from streamlit_agraph import Node, Edge, Config, TripleStore, agraph
import networkx as nx
import textwrap
def remove_files():
    if os.path.isdir("data"):
        files = glob.glob('data/*')
        for f in files:
            os.remove(f)
    else:
        os.mkdir("data")

    return 1

def savefile(uploaded_file):
    try:
        if remove_files() ==1:
            with open(os.path.join("data", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            return 1
        else:
            return 0
    except Exception as ex:
        st.write(f"Error {ex} while uploading file: {uploaded_file}")

def read_file(uploaded_file, file_details):
    dataframe = None
    if (uploaded_file.name.split(".")[1] == "hdf"):
        dataframe = pd.read_hdf(f"data/{file_details['name']}")
    elif (uploaded_file.name.split(".")[1] == "csv"):
        dataframe = pd.read_csv(f"data/{file_details['name']}")

    return dataframe

def select_algorithm():
    option = st.selectbox(
        label='Select Optimisation Algorithm',
        key='algorithm_select',
        options=('Select Algorithm', 'Notears'),
    )

    return option

def slider(label,min_value, max_value, defaultselected, step, key):
    threshold = st.slider(
        label=label,
        min_value=min_value, max_value=max_value, value=defaultselected,step=step, key=key)
    return threshold

def multiselect(options, key, text=""):
    return  st.multiselect(
        key=key,
        label=text,
     options=options)

def start_structure_learning(text,key):
    return st.button(label=text,key=key)

def is_networkxgraph(graph):
    isNx = (type(graph) == nx.classes.multidigraph.MultiDiGraph or type(graph) == nx.classes.multidigraph.DiGraph)
    return isNx

def convert_agraph_node(graph):
    """ Converts the networx graph nodes to agraph nodes
    :param graph: network graph
    :return: agraph Node
    """
    _nodes = []
    if is_networkxgraph(graph):
        for nodelist in list(graph.nodes()):
            _nodes.append(Node(id=nodelist,
                               size=400,
                               ))
    return _nodes


def zero_error():
    return "Error occurred. It could either be due to dataset or the configurations are too strong that model didn't " \
           "learn or was filtered out. Try adjusting the configuration parameters or checking dataset."

def invalid_selection():
    return "Invalid selection. Source node should be greater than or equal to destination."

def convert_agraph_edge(graph):
    """Converts the networx graph edges to agraph edges
    :param graph:  network graph
    :return: agraph Edge
    """

    _edges = []
    if is_networkxgraph(graph):

        for edgelist in list(graph.edges()):
            s, t = edgelist
            _edges.append(Edge(source=s,
                               target=t,
                               type="CURVE_SMOOTH"))

    return _edges

def gaph_config():
    """ Set the configuration to the interactive window for DAG network visualisation
    :return: Config
    """
    config = Config(width=900,
                    height=1000,
                    directed=True,
                    nodeHighlightBehavior=True,
                    highlightColor="#F7A7A6",  # or "blue"
                    collapsible=True,
                    node={'labelProperty': 'label'},
                    link={'labelProperty': 'label', 'renderLabel': False}
                    )
    return config

@st.cache
def convert_df(df):
    return  df.to_csv().encode('utf-8')


def preprocess_data(df):
    """Standardise and impute the values of the dataframe
    :param df: input pandas dataframe
    :return: preprocessed dataframe
    """
    from sklearn import preprocessing
    from sklearn.impute import SimpleImputer
    column_name = list(df.columns)
    imp = SimpleImputer().fit_transform(df)
    standard_impute = preprocessing.MinMaxScaler().fit_transform(imp)
    return pd.DataFrame(standard_impute, columns=column_name)

def checkbox(label, key):
    return st.checkbox(label=label, key=key)

def check_hidden_layer_input(input_text):
    if type(input_text) == str:
        strlist = input_text.split(",")
        check_res = [int(ip) if ip.isdigit() else False for ip in strlist]
        return None if False in check_res else check_res
    else:
        return None

def get_owl_file():
    return "generated_ontology.owl"

def sparql_prefix():
    prefix = textwrap.dedent(""" 
               PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
           """)
    return prefix
def get_all_obj_properties():
    graph = Graph()
    graph.parse(get_owl_file())
    q = """
    SELECT   ?o ?p  
               WHERE { ?s rdf:type owl:NamedIndividual ;
               ?p ?o. 
               FILTER regex(str(?p), "isInfluencedBy")
               }
               """
    qres = graph.query(q)
    return pd.DataFrame(tuple(set([p.split("#")[1] for o, p in qres])))




def get_all_data_properties_decimals():
    graph = Graph()
    graph.parse(get_owl_file())
    q = """ 
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
             SELECT   ?o ?p  
               WHERE { ?s rdf:type owl:NamedIndividual ;
               ?p ?o. 
               FILTER (datatype(?o)=xsd:decimal)
               }
            """
    qres = graph.query(q)
    return pd.DataFrame(tuple(set([o for o, p in qres])))



def filter_by_obj_property(prop_name):
    graph = Graph()
    graph.parse(get_owl_file())
    q = textwrap.dedent("""
            {0}
           SELECT  ?o  ?p ?s 
           WHERE {{ ?s rdf:type owl:NamedIndividual ;
           ?p ?o. 
           FILTER(?p = :{1})
           }}
        """).format(sparql_prefix(), prop_name)
    tripes = graph.query(q)
    return tripes



def filter_by_data_property_value(dp_value):
    graph = Graph()
    graph.parse(get_owl_file())
    q = textwrap.dedent("""
            {0}
           SELECT  ?o  ?p ?s 
           WHERE {{ ?s rdf:type owl:NamedIndividual ;
           ?p ?o. 
           FILTER(?o >= {1})
           }}
        """).format(sparql_prefix(), dp_value)
    tripes = graph.query(q)
    return tripes

def filter_by_data_property_value_type(dp_value):
    graph = Graph()
    graph.parse(get_owl_file())
    q = textwrap.dedent("""  SELECT  ?o  ?p ?s 
           WHERE {{ ?s rdf:type owl:NamedIndividual ;
           ?p ?o. 
           FILTER(?o ='{0}')
           }}
        """).format(dp_value)
    print(q)
    tripes = graph.query(q)
    return tripes

def visualize_triples(triples):
    config = Config(width=900,
                    height=900,
                    directed=True,
                    nodeHighlightBehavior=True,
                    highlightColor="#F7A7A6",  # or "blue"
                    collapsible=True,
                    node={'labelProperty': 'label'},
                    link={'labelProperty': 'label', 'renderLabel': True}
                    )

    store = TripleStore()

    for subj, pred, obj in triples:
        store.add_triple(subj, pred, obj, "")

    return agraph(list(store.getNodes()), list(store.getEdges()), config)




