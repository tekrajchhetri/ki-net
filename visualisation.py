# -*- coding: utf-8 -*-
# @Time    : 18.03.22 15:26
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : https://tekrajchhetri.com/
# @File    : visualisation.py
# @Software: PyCharm

import numpy as np
from causalnex.structure import StructureModel
from causalnex.plots import plot_structure
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE

def visualise_linear(sm, threshold):
    sm.remove_edges_below_threshold(threshold)
    viz = plot_structure(
        sm,
        graph_attributes={"scale": "0.5"},
        all_node_attributes=NODE_STYLE.WEAK,
        all_edge_attributes=EDGE_STYLE.WEAK,
    )
    mpl.rcParams["figure.dpi"] = 120
    # fig = plt.figure(figsize=(15, 8))  # set figsize
    return nx.draw_networkx(viz)