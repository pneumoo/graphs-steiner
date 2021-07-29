# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 12:47:35 2015
http://gographer.googlecode.com/svn-history/r56/trunk/SteinerTree.py
@author: Brian
"""

# This is a generalized implementation of the Kou algorithm for creating Steiner Trees.  It is not
# tied to GOGrapher and can be used with any networkx wieghted graph.

from heapq import *
from networkx import *
from networkx import Graph

## Extract a Steiner tree from a weighted graph, given a list of vertices of interest
# @param G  A Graph with weighted edges
# @param voi  A list of vertices of interest
# @param generator A method to make a new Graph instance (in the case that you've extended Graph)
# \returns a new graph if no errors, None otherwise
def make_steiner_tree(G, voi, generator=None):
        mst = Graph()
        for v in voi:
                if not v in G:
                        raise ValueError("make_steiner_tree(): Some vertice not in original graph")
        if len(voi) == 0:
                return mst
        if len(voi) == 1:
                mst.add_node(voi[0])
                return mst

        # Initially, use (a version of) Kruskal's algorithm to extract a minimal spanning tree
        # from a weighted graph.  This algorithm differs in that only a subset of vertices are
        # going to be present in the final subgraph (which is not truely a MST - must use Prim's
        # algorithm later.

        # extract all shortest paths among the voi
        heapq = []
        paths = {}

        # load all the paths bwteen the Steiner vertices. Store them in a heap queue
        # and reconstruct the MST of the complete graph using Kruskal's algorithm
        for i in range(len(voi) - 1):
                v1 = voi[i]
                for v2  in voi[i+1:]:
                        result = bidirectional_dijkstra(G, v1, v2)
                        if result == False:
                                raise RuntimeError("The two vertices given (%s, %s) don't exist on the same connected graph" % (v1, v2))
                                #print "The two vertices given (%s, %s) don't exist on the same connected graph" % (v1, v2)
                        distance, vertList = result
                        keys = [v1, v2]
                        keys.sort()
                        key = "%s:%s" % tuple(keys)
                        paths[key] = (vertList)
                        heappush(heapq, (distance, v1, v2))

                                
        # construct the minimum spanning tree of the complete graph
        while heapq:
                w, v1, v2 = heappop(heapq)
                # if no path exists yet between v1 and v2, add this one
                if v1 not in mst or v2 not in mst or not has_path(mst, v1, v2):
                        mst.add_edge(v1, v2,weight=w)

        # check if the graph is tree and correct
        sTree = set(mst.nodes())
        sSteiner = set(voi)
        if sTree ^ sSteiner:
                raise RuntimeError('Failed to construct MST spanning tree')
        
        # reconstruct subgraph of origGraph using the paths
        if generator is None:
                subgraph  = Graph()
        else:
                subgraph = generator()
        for edge in mst.edges_iter(data=True):
                keys = [edge[0],edge[1]]
                keys.sort()
                key = "%s:%s" % tuple(keys)
                vList = paths[key]
                for i in range(len(vList) - 1):
                        v1 = vList[i]
                        v2 = vList[i+1]
                        w = G[v1][v2]
                        subgraph.add_edge(v1, v2, w)
        # get rid of possible loops - result will be a true MST
        subgraph = make_prim_mst(subgraph, generator)

        # remove intermediate nodes in paths that are not in list of voi
        return _trimTree(subgraph, voi)


# remove intermediate nodes in paths that are not in list of voi in given graph
def _trimTree(graph, voi):
        trimKeepTrack = []
        firstNode = voi[0]
        if len(graph.neighbors(firstNode)) < 2:
                trimKeepTrack.append(firstNode)
                firstNeighbor = graph.neighbors(firstNode)[0]
                trimKeepTrack.append(firstNeighbor)
                graph = _trim(firstNeighbor, graph, trimKeepTrack, voi)
        else:
                trimKeepTrack.append(firstNode)
                graph = _trim(firstNode, graph, trimKeepTrack, voi)
        return graph


def _trim(node, graph, trimKeepTrack, voi):
        if len(graph.adj[node].keys()) > 1:
                for nodeNeighbor in graph.adj[node].keys():
                        if nodeNeighbor not in trimKeepTrack:
                                trimKeepTrack.append(nodeNeighbor)
                                graph = _trim(nodeNeighbor, graph, trimKeepTrack, voi)
        if len(graph.adj[node].keys()) < 2:
                if node not in voi:
                        graph.remove_node(node)
        return graph


"""
Prim's algorithm: constructs the minimum spanning tree (MST) from an instance of weighted Graph
@param G An weighted Graph()
@param generator A method to make a new Graph instance (in the case that you've extended Graph)
\returns A MST verstion of G
"""
def make_prim_mst(G, generator=None):
	if generator is None:
		mst = Graph()
	else:
		mst = generator()       
	#priorityQ is a list of list (the reverse of the edge tuple with the weight in the front)
	priorityQ = []
	firstNode = G.nodes()[0]
	mst.add_node(firstNode)
	for edge in G.edges_iter(firstNode, data=True):
		if len(edge) != 3 or edge[2] is None:
			raise ValueError("make_prim_mst accepts a weighted graph only (with numerical weights)")
		heappush(priorityQ, (edge[2], edge))

	while len(mst.edges()) < (G.order()-1):
		w, minEdge = heappop(priorityQ)
		if len(minEdge) != 3 or minEdge[2] is None:
			raise ValueError("make_prim_mst accepts a weighted graph only (with numerical weights)")
		v1, v2, w = minEdge
		if v1 not in mst:
			for edge in G.edges_iter(v1, data=True):
				if edge == minEdge:
					continue
				heappush(priorityQ, (edge[2], edge))
		elif v2 not in mst:
			for edge in G.edges_iter(v2, data=True):
				if edge == minEdge:
					continue
				heappush(priorityQ, (edge[2], edge))
		else:
			# non-crossing edge 
			continue 
		mst.add_edge(minEdge[0],minEdge[1],minEdge[2])
	return mst
 
 
 
def test():
    #~~~~~~~~~~~~~~~~~~~~~~~~~DO NOT DELETE THIS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Add a set of X,Y coordinates
    x = [1, 1, 2, 2]
    y = [1, 2, 1, 2]
    
    G = nx.Graph()
    G.add_nodes_from(range(len(x)))     #Add qty nodes = length x
    #nx.set_node_attributes(G,'pos',pos)
    for n in G:
        G.node[n]['pos']=(x[n],y[n])    #store x,y coordinates for each node
    
    nodes = G.nodes(data=True)
    while nodes:
        u,du = nodes.pop()
        pu = du['pos']
        for v,dv in nodes:
            pv = dv['pos']
            e=(u,v) #The edge we're looking at
            r = (sum(((a-b)**2 for a,b in zip(pu,pv))))**0.5    #weight of edge
            G.add_edge(u,v,weight=r)
    
  
    Gpos = get_node_attributes(G,'pos')
    print("Node X,Y Positions:\n", list(get_node_attributes(G,'pos').values()))
    print("Lengths Between Nodes:\n", get_edge_attributes(G,'weight'))
    nx.draw_networkx_edges(G,Gpos)
    nx.draw_networkx_nodes(G,Gpos,node_size=100)


    #STEINER ON THIS BITCH
    voi=[0,1,2,3]
    S=make_steiner_tree(G,voi)    

    Spos = get_node_attributes(G,'pos')
    print("S Node X,Y Positions:\n", list(get_node_attributes(S,'pos').values()))
    print("S Lengths Between Nodes:\n", get_edge_attributes(S,'weight'))
    nx.draw_networkx_edges(S,Spos)
    nx.draw_networkx_nodes(S,Spos,node_size=100)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 
if __name__ == "__main__":
    test()