# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 07:30:18 2015

@author: Brian
"""
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
X = csr_matrix([[0, 8, 0, 3],[0, 0, 2, 5],[0, 0, 0, 6],[0, 0, 0, 0]])
Tcsr = minimum_spanning_tree(X)
Tcsr.toarray().astype(int)

