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
from streamlit_agraph import Node, Edge, Config
import networkx as nx

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
    return "Error occurred. Check your dataset."

def invalid_selection():
    return "Number of nodes in source and destination should be equal."

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
    config = Config(width=600,
                    height=800,
                    directed=True,
                    nodeHighlightBehavior=True,
                    highlightColor="#F7A7A6",  # or "blue"
                    collapsible=True,
                    node={'labelProperty': 'label'},
                    link={'labelProperty': 'label', 'renderLabel': False}
                    )
    return config



def checkbox(label, key):
    return st.checkbox(label=label, key=key)

def check_hidden_layer_input(input_text):
    if type(input_text) == str:
        strlist = input_text.split(",")
        check_res = [int(ip) if ip.isdigit() else False for ip in strlist]
        return None if False in check_res else check_res
    else:
        return None





