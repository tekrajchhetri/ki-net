# -*- coding: utf-8 -*-
# @Time    : 18.03.22 15:26
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : https://tekrajchhetri.com/
# @File    : visualisation.py
# @Software: PyCharm

from causalnex.plots import plot_structure
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st
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

def display_logo():
    st.markdown(
       body= f"""
        <p float="left">
    <img src="https://www.sti-innsbruck.at/sites/default/files/uploads/media/STI-IBK-Logo_CMYK_Pfad_XL.jpg" alt="STI Innsbruck" width="300px"/>
    <img src="https://upload.wikimedia.org/wikipedia/de/thumb/d/dc/Hs-kempten-logo.svg/602px-Hs-kempten-logo.svg.png" alt="HS-Kempten" width="150px"/> 
    <img src="https://www.uar.at/files/assets/content/Logos/SCCH_Logo_Subline_.jpg" width="240px"  alt="SCCH"/>
    </p>
        """, unsafe_allow_html =True
    )