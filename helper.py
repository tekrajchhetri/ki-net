# -*- coding: utf-8 -*-
# @Time    : 18.03.22 14:40
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : https://tekrajchhetri.com/
# @File    : helper.py
# @Software: PyCharm

import os
import glob

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