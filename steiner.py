# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 19:42:11 2015

@author: Brian

"""

from scipy import *
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Plot Star from mean location
def pt2pts_graph(N,pt):
    l = len(N)
    for i in range(l):
        plt.plot([pt[0],N[i,0]],[pt[1],N[i,1]],'r')

# Make a complete graph
def complete_graph(N):
    l = len(N)
    for i in range(l):
        for j in range(l):
            plt.plot([N[i,0],N[j,0]],[N[i,1],N[j,1]],'b')



nodes = 10   #number of nodes to be connected by tree
N = np.random.uniform(size=(nodes, 2))
mean = [(sum(N[:,0])/nodes), (sum(N[:,1])/nodes)]
distmat = squareform(pdist(N))


plt.scatter(N[:,0], N[:,1])
plt.axis([0,1,0,1])
plt.scatter(mean[0],mean[1],marker='x') 
#plt.plot(N[:,0], N[:,1])

pt2pts_graph(N,mean)
complete_graph(N)


    
    
    
#=============================================================================    

'''Creates a complete graph of edge weights in E'''
def funtest(E, copy_V=True):
    
    #Function Housekeeping ---
    if copy_E:      #Creates a local copy of E
        E = E.copy()    
    if E.shape[0] != E.shape[1]:    #Checks that E is square
        raise ValueError("E must be a sq matrix of edge weights.")
    
    #Intitialization ---
    n_vertices = E.shape[0]     #Counts number of vertices        
    edges = []                  #Pre allocates matrix for edges
    visited_vertices = [0]      #Sets to 1st vertext for use in loop
    num_visited = 1             #Count of how many we've looked at
    np.fill_diagonal(E,np.inf)  #Fills diagonal w/ "inf" so a vertex can't connect w/ itself
    
    #Iterating through matrix ---
    while num_visited != n_vertices:
        new_edge = 1
    

def minimum_spanning_tree(X, copy_X=True):
    """X are edge weights of fully connected graph"""
    if copy_X:
        X = X.copy()
 
    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")

    n_vertices = X.shape[0]
    spanning_edges = []
     
    # initialize with node 0:                                                                                         
    visited_vertices = [0]                                                                                            
    num_visited = 1
    # exclude self connections:
    diag_indices = np.arange(n_vertices)

    X[diag_indices, diag_indices] = np.inf

    
    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)

        # 2d encoding of new_edge from flat, get correct indices                                                      
        new_edge = divmod(new_edge, n_vertices)

        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]  
                                                   
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf
                                                                   
        num_visited += 1
    return np.vstack(spanning_edges)


'''
# testing to get lines to plot from MEAN to all Nodes
for i in range(nodes):
    plt.plot([mean[0], mean[1]], [1,1], c='r')
'''


"""
# find 3 closest points to mean x,y
D=[]    # array of distances
for i in range(nodes):
    dx = X[i] - meanx
    dy = Y[i] - meany
    dist = (dx**2 + dy**2)**0.5

    D.append(dist)

"""
