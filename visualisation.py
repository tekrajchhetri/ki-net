# -*- coding: utf-8 -*-
# @Time    : 18.03.22 15:26
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : https://tekrajchhetri.com/
# @File    : visualisation.py
# @Software: PyCharm

import streamlit as st


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


