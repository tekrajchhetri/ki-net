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
        options=('Select Algorithm', 'NotearsLinear', 'NotearsMLP'),
    )

    return option

def threshold_param():
    threshold = st.slider(
        'Select filter threshold %',
        0.0, 1.0, 0.4)
    return threshold

def start_structure_learning():
    return st.button("Start Bayesian Structure Learning Process")

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
    config = Config(width=500,
                    height=500,
                    directed=True,
                    nodeHighlightBehavior=True,
                    highlightColor="#F7A7A6",  # or "blue"
                    collapsible=True,
                    node={'labelProperty': 'label'},
                    link={'labelProperty': 'label', 'renderLabel': False}
                    )
    return config

