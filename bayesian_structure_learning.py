# -*- coding: utf-8 -*-
# @Time    : 18.03.22 14:46
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : https://tekrajchhetri.com/
# @File    : bayesian_structure_learning.py
# @Software: PyCharm

from causalnex.structure.pytorch.notears import from_pandas

def start_linear_structure_learning(dataset):
    sm = from_pandas(dataset)
    return sm