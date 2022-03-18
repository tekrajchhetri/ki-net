from io import StringIO

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob

st.title('Optimising Manufacturing Process with Bayesian Learning and Knowledge Graphs')

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

def file_upload():
    uploaded_file = st.file_uploader("Upload your file in CSV or HDF format")
    if uploaded_file is not None:
        saved_file = savefile(uploaded_file)
        print(saved_file)
        file_details = {"name": uploaded_file.name, "type": uploaded_file.type}
        try:
            if (uploaded_file.name.split(".")[1] == "hdf"):
                dataframe = pd.read_hdf(f"data/{file_details['name']}")
            elif (uploaded_file.name.split(".")[1] == "csv"):
                dataframe = pd.read_csv(f"data/{file_details['name']}")
            st.write(dataframe)
        except Exception as e:
            st.write(e)
            st.write("Invalid file format")


if __name__ == '__main__':
    remove_files()
    file_upload()