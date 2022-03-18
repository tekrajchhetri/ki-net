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