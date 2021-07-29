__author__ = 'Samuel'
 
import random
import math
from itertools import combinations
import numpy as np
import pylab as pl
from matplotlib import collections as mc
from math import sin, cos, pi, factorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
def dist(a, b):
    """Computes the distance between two points"""
    return (sum([(a[i] - b[i]) ** 2 for i in range(len(a))]) ** .5)
 
def unique(lis):
    """determines whether all the elements of a list are unique"""
    return list(set(lis)) == sorted(lis)
 
def hasNoAlmostPairs(lis):
    """forces trees to not have overlapping steiner points"""
    lx = []
    ly = []
    for i in lis:
        for j in range(len(lx)):
            if abs(i[0]-lx[j]) < .0001 and abs(i[1]-ly[j]) < .0001:
                return False
        lx.append(i[0])
        ly.append(i[1])
    return True
 
def createTree(snConnections, ssConnections):
    """Finds the tree structure denoted by snConnections (between steiner points
     and nodes) and ssConnections (connections between steiner points and other steiner points"""
    allPoints = [i for i in range(len(snConnections) - 2)] + [i + len(snConnections) - 2 for i in
                                                              range(len(snConnections))]
    tree = [(snConnections[i], i + len(snConnections) - 2) for i in range(len(snConnections))]
    allssConnections = []
    for i in combinations([i for i in range(len(snConnections) - 2)], 2):
        allssConnections.append(i)
    for i in ssConnections:
        tree.append(allssConnections[i])
    return tree
 
def DFS(tree):
    """Depth-first search"""
    stack = []
    discovered = []
    stack.insert(0, tree[0][0])
    while stack:
        v = stack[-1]
        stack = stack[:-1]
        if v not in discovered:
            discovered.append(v)
            for edge in tree:
                if v in edge:
                    for point in edge:
                        if point != v:
                            stack.insert(0, point)
    return discovered
 
def isFullyAttached(tree):
    if len(DFS(tree)) == len(set.union(set([i[0] for i in tree]), set([i[1] for i in tree]))):
        return True
    return False
 
def isValid(snConnections, ssConnections, steinerPoints):
    #Checks 3 things:
    # 1. whether the graph is a tree (acyclic, connected)
    # 2. whether each steiner point has degree 3
    # 3. there are no overlapping steiner points
    tree = createTree(snConnections, ssConnections)
    if not isFullyAttached(tree) or not hasNoAlmostPairs(steinerPoints):
        return False
 
    steinerPointsCounts = [0 for i in range(len(snConnections) - 2)]
    for i in snConnections:
        steinerPointsCounts[i] += 1
    allssConnections = []
    for i in combinations([i for i in range(len(snConnections) - 2)], 2):
        allssConnections.append(i)
    for i in ssConnections:
        for j in allssConnections[i]:
            steinerPointsCounts[j] += 1
    if all([i == 3 for i in steinerPointsCounts]):
        return True
    return False
 
def normalize(vec):
    """converts a vector into a unit vector"""
    return [i/(sum([j ** 2 for j in vec]) ** .5) for i in vec]
 
def nCr(n, r):
    if n < r:
        return 0
    return factorial(n) / (factorial(r) * factorial(n - r))
 
def pathLength(nodes, steinerPoints, ssConnections, snConnections):
    """computes the length of the current tree"""
    n = len(nodes)
    d = 0
 
    d += sum([dist(nodes[i], steinerPoints[snConnections[i]]) for i in range(n)])
    # allssConnections = []
    # for i in combinations([i for i in range(n - 2)], 2):
    #     allssConnections.append(i)
    allssConnections = [j for j in combinations([i for i in range(n - 2)], 2)]
 
    d += sum([dist(steinerPoints[allssConnections[i][0]],
              steinerPoints[allssConnections[i][1]]) for i in ssConnections])
    return d
 
def steinerTree(nodes):
    """Computes the Steiner Tree using an iterative heuristic"""
    #works in 2 or 3 dimensions
    R = len(nodes[0]) # either 2 or 3 -- this is the dimension we're working in
    n = len(nodes)
    steinerPoints = []
    for i in range(n - 2):
        steinerPoints.append([random.uniform(min([i[dim] for i in nodes]), max([i[dim] for i in nodes])) for dim in
                     range(R)])
    jump = 0
    for i in steinerPoints:
        for j in nodes:
            jump += dist(i, j)
    jump /= (len(steinerPoints) * len(nodes))
    #now the initial topology must be created
    snLocs = [i for i in range(n - 2)]
    snConnections = [random.choice(snLocs) for i in range(len(nodes))] #connections between steiner points and nodes
    ssLocs = [i for i in range(int(nCr(len(steinerPoints), 2)))]
    ssConnections = []  #connections between steiner points and other steiner points
    for i in range(n - 3):
        ssConnections.append(random.choice(ssLocs))
        ssLocs.remove(ssConnections[-1])
    print(createTree(snConnections, ssConnections))  #this is the structure of the initial tree
    iterations = 0
    while iterations < 25000:
        oldConnections = (snConnections[:],
                          ssConnections[:])  #these fucking colons needing to be here cost me hours of time
 
        vec = [random.random() for dim in range(R)]
        negaters = [random.randint(0, 1) for dim in range(R)]
        for dim in range(R):
            if negaters[dim]:
                vec[dim] *= -1
        vec = normalize(vec)
        #multiply each component by the jump size
        for j in range(R):
            vec[j] *= jump
        r = random.randint(0, len(steinerPoints) - 1)
        newsol = [steinerPoints[r][dim] + vec[dim] for dim in range(R)]
        newsteinerPoints = steinerPoints[:r] + [newsol] + steinerPoints[r + 1:]
        if pathLength(nodes, newsteinerPoints, ssConnections, snConnections) < \
                pathLength(nodes, steinerPoints, ssConnections, snConnections):
            steinerPoints = newsteinerPoints
 
        r1 = random.randint(0, len(snConnections) - 1)
        r2 = random.randint(0, len(snConnections) - 1)
        newSnConnections = snConnections[:]
        newSnConnections[r1], newSnConnections[r2] = newSnConnections[r2], newSnConnections[r1]
        if pathLength(nodes, steinerPoints, ssConnections, newSnConnections) < \
                pathLength(nodes, steinerPoints, ssConnections,snConnections):
            snConnections = newSnConnections[:]
        r = random.randint(0, len(ssConnections) - 1)
        newSsConnection = random.randint(0, nCr(len(steinerPoints), 2) - 1)
        if pathLength(nodes, steinerPoints, ssConnections[:r] + [newSsConnection] + ssConnections[r + 1:], snConnections) < \
                pathLength(nodes, steinerPoints, ssConnections, snConnections) and unique(
                                ssConnections[:r] + [newSsConnection] + ssConnections[r + 1:]):
            ssConnections[r] = newSsConnection
            allssConnections = [i for i in combinations([i for i in range(n - 2)], 2)]
            steinerPointsCounts = [3 for i in range(len(steinerPoints))]
            for i in ssConnections:
                for j in allssConnections[i]:
                    steinerPointsCounts[j] -= 1
            snConnections = []
            for i in range(len(steinerPointsCounts)):
                for j in range(steinerPointsCounts[i]):
                    snConnections.append(i)
            random.shuffle(snConnections)
        if not isValid(snConnections, ssConnections, steinerPoints):
            snConnections, ssConnections = oldConnections
        jump *= .9995
        iterations += 1
        if iterations == 25000 and not isValid(snConnections, ssConnections, steinerPoints):
            # restarts if we've failed
            print("Starting over...")
            steinerPoints = []
            for i in range(n - 2):
                steinerPoints.append([random.uniform(min([i[dim] for i in nodes]), max([i[dim] for i in nodes])) for dim in
                             range(R)])
            jump = 0
            for i in steinerPoints:
                for j in nodes:
                    jump += dist(i, j)
            jump /= (len(steinerPoints) * len(nodes))
            #now the initial topology must be created
            snLocs = [i for i in range(n - 2)]
            snConnections = [random.choice(snLocs) for i in range(len(nodes))] #connections between steiner points and nodes
            ssLocs = [i for i in range(int(nCr(len(steinerPoints), 2)))]
            ssConnections = []  #connections between steiner points and other steiner points
            for i in range(n - 3):
                ssConnections.append(random.choice(ssLocs))
                ssLocs.remove(ssConnections[-1])
            iterations = 0
 
    #wrap up program
    
    print("steinerPoints:")
    for sol in steinerPoints:
        print(sol)
    print("ssConnections: ", ssConnections)
    print("snConnections: ", snConnections)
    print("tree: ", createTree(snConnections, ssConnections))
    print(pathLength(nodes, steinerPoints, ssConnections, snConnections))
    # if not isValid(snConnections, ssConnections):
    #     print("I have not generated a valid Steiner tree for you. I am very sorry.")
    #     return
 
    #for 2D plots:
    if R == 2:
        lines = []
        for i in range(n):
            lines.append([nodes[i], steinerPoints[snConnections[i]]])
        allssConnections = []
        for i in combinations([i for i in range(n - 2)], 2):
            allssConnections.append(i)
        for i in ssConnections:
            lines.append([steinerPoints[allssConnections[i][0]], steinerPoints[allssConnections[i][1]]])
        col = np.array([(0, 0, 0, 1)])
        lc = mc.LineCollection(lines, colors=col, linewidths=2)
        fig, ax = pl.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        pl.plot([i[0] for i in steinerPoints], [i[1] for i in steinerPoints], 'bo')
        pl.plot([i[0] for i in nodes], [i[1] for i in nodes], 'ro')
        pl.axis([min(i[0] for i in nodes) - 1.5, max(i[0] for i in nodes) + 1.5,
                 min(i[1] for i in nodes) - 1.5, max(i[1] for i in nodes) + 1.5])
        ax.text(min([i[0] for i in nodes]) - 1, min(i[1] for i in nodes) - 1,
                "Total distance: " + str(pathLength(nodes, steinerPoints, ssConnections, snConnections)), fontsize=15)
        pl.show()
 
    #for 3D plots
    if R == 3:
        lines = []
        for i in range(n):
            lines.append([nodes[i], steinerPoints[snConnections[i]]])
        allssConnections = []
        for i in combinations([i for i in range(n - 2)], 2):
            allssConnections.append(i)
        for i in ssConnections:
            lines.append([steinerPoints[allssConnections[i][0]], steinerPoints[allssConnections[i][1]]])
        VecStart_x = []
        VecStart_y = []
        VecStart_z = []
        VecEnd_x = []
        VecEnd_y = []
        VecEnd_z = []
        for line in lines:
            VecStart_x.append(line[0][0])
            VecEnd_x.append(line[1][0])
            VecStart_y.append(line[0][1])
            VecEnd_y.append(line[1][1])
            VecStart_z.append(line[0][2])
            VecEnd_z.append(line[1][2])
 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(VecStart_x)):
            ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i], VecEnd_y[i]], zs=[VecStart_z[i], VecEnd_z[i]])
        pl.plot([i[0] for i in steinerPoints], [i[1] for i in steinerPoints], [i[2] for i in steinerPoints], 'bo')
        pl.plot([i[0] for i in nodes], [i[1] for i in nodes], [i[2] for i in nodes], 'ro')
        # ax.text(min([i[0] for i in nodes])-1, min(i[1] for i in nodes)-1, min(i[2] for i in nodes)-1,
        #         "Total distance: "+str(pathLength(nodes, steinerPoints, ssConnections, snConnections)), fontsize=15)
        ax.set_title("Total Distance: " + str(pathLength(nodes, steinerPoints, ssConnections, snConnections)))
        pl.show()
 
#NOTE: the more points, the lower the likelihood of working well
#run the program several times instead of just once
 
#if the user is to give input
# points0 = []
# n = int(input("How many nodes? "))
# for i in range(n):
#     point = input("Enter a node (x,y[, z]): ")
#     points0.append(point)
# points1 = []
# for i in points0:
#     s = ''
#     for j in i:
#         if j not in '( )':
#             s += j
#     points1.append(i)
# points2 = []
# for i in points1:
#     commaLoc = i.index(',')
#     p = (float(i[:commaLoc]),float(i[commaLoc+1:]))
#     points2.append(p)
# nodes = points2
 
#some good examples to try
#uncomment whichever you want
 
def steinerTreePolygon(numSides):
    nodes = []
    n = numSides
    for i in range(n):
        nodes.append((cos(2 * pi * (i / n)), sin(2 * pi * (i / n))))
    steinerTree(nodes)
#steinerTreePolygon(4)
 
#random nodes (2D)
#n = 8
# nodes = [(random.uniform(0,5), random.uniform(0,5)) for i in range(n)]
# steinerTree(nodes)
 
#random nodes (3D)
#n = 6
# nodes = [(random.uniform(0,5), random.uniform(0,5), random.uniform(0,5)) for i in range(n)]
# steinerTree(nodes)
 
# 3 x 3 grid
#steinerTree(((0,0),(3,0),(6,0),(0,3),(3,3),(6,3),(6,6),(3,6),(0,6)))
 
#eight points arranged in a parallelogram 4x2
steinerTree(((0,0),(2,0),(1,1),(3,1),(2,2),(4,2),(3,3),(5,3)))
 
#10 points arranged in a parallelogram 5x2
#steinerTree(((0,0),(2,0),(1,1),(3,1),(2,2),(4,2),(3,3),(5,3),(4,4),(6,4)))
 
#12 points arranged in a parallelogram 6x2
#steinerTree(((0,0),(2,0),(1,1),(3,1),(2,2),(4,2),(3,3),(5,3),(4,4),(6,4),(5,5),(7,5)))
 
#eight points in a rhombus
# steinerTree(((0,0),(4,-1),(4,1),(8,-2),(8,2),(12,-1),(12,1),(16,0)))
 
#nine points in a parallelogram 3x3
#steinerTree(((0,0),(3,0),(6,0),(1,2),(4,2),(7,2),(2,4),(5,4),(8,4)))
 
##ten points in a pentagram
# nodes = []
# n = 5
# for i in range(n):
#         nodes.append([cos(2*pi*(i/n)),sin(2*pi*(i/n))])
# for i in ([-0.809016994374947, 2.4898982848827798],\
#           [2.118033988749895, -1.5388417685876268],\
#           [-2.618033988749896, 0],\
#           [-0.8090169943749479, -2.4898982848827806],\
#           [2.118033988749895, 1.5388417685876266]):
#     nodes.append(i)
# steinerTree(nodes)
 
#ten points in a pentagram, and the origin (11 total)
# nodes = []
# n = 5
# for i in range(n):
#         nodes.append([cos(2*pi*(i/n)),sin(2*pi*(i/n))])
# for i in ([-0.809016994374947, 2.4898982848827798],\
#           [2.118033988749895, -1.5388417685876268],\
#           [-2.618033988749896, 0],\
#           [-0.8090169943749479, -2.4898982848827806],\
#           [2.118033988749895, 1.5388417685876266],\
#           [0,0]):
#     nodes.append(i)
# steinerTree(nodes)
 
##12 points arranged in a hexagram
# nodes = []
# n = 6
# for i in range(n):
#         nodes.append([cos(2*pi*(i/n)),sin(2*pi*(i/n))])
# for i in ([3**.5 * cos(pi/6), 3**.5 * sin(pi/6)],
#           [0, 3**.5],
#           [-(3**.5 * cos(pi/6)), 3**.5 * sin(pi/6)],
#           [-(3**.5 * cos(pi/6)), -(3**.5 * sin(pi/6))],
#           [0, -(3**.5)],
#           [3**.5 * cos(pi/6), -(3**.5 * sin(pi/6))]):
#     nodes.append(i)
# steinerTree(nodes)
 
##12 points arranged in a hexagram, with the origin (13 total)
# nodes = []
# n = 6
# for i in range(n):
#         nodes.append([cos(2*pi*(i/n)),sin(2*pi*(i/n))])
# for i in ([3**.5 * cos(pi/6), 3**.5 * sin(pi/6)],
#           [0, 3**.5],
#           [-(3**.5 * cos(pi/6)), 3**.5 * sin(pi/6)],
#           [-(3**.5 * cos(pi/6)), -(3**.5 * sin(pi/6))],
#           [0, -(3**.5)],
#           [3**.5 * cos(pi/6), -(3**.5 * sin(pi/6))],
#           [0,0]):
#     nodes.append(i)
# steinerTree(nodes)
 
##12 points arranged in a right triangle
#steinerTree(((0,0),(1,0),(2,0),(3,0),(4,0),(4,1),(4,2),(4,3),(4,4),(1,1),(2,2),(3,3)))
 
##12 points arranged in an equilateral triangle
# steinerTree(((-2,0),(-1,0),(0,0),(1,0),(2,0),(0,2*3**.5),\
#              (.5,1.5*3**.5),(1,3**.5),(1.5,.5*3**.5),\
#              (-.5,1.5*3**.5),(-1,3**.5),(-1.5,.5*3**.5)))
 
##12 points in a thick plus sign
#steinerTree(((1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)))
 
#8 points  arranged in a cube (3D)
#steinerTree(((0,0,0),(1,0,0),(0,1,0),(0,0,1),(0,1,1),(1,0,1),(1,1,0),(1,1,1)))
#solution:
# steinerPoints:
# [0.7113248710539583, 0.500000043151846, 0.49999999640310383]
# [0.8556624539373641, 0.5000000394786789, 0.24999998062671402]
# [0.14433756974080342, 0.50000001900439, 0.7499999881138082]
# [0.8556624301183892, 0.500000035562603, 0.7499999901123915]
# [0.28867512066534423, 0.5000000263639303, 0.5000000025402277]
# [0.14433755846316024, 0.500000022234928, 0.2500000054998505]
# ssConnections:  [2, 3, 10, 0, 14]
# snConnections:  [5, 1, 5, 2, 2, 3, 1, 3]
# tree:  [(5, 6), (1, 7), (5, 8), (2, 9), (2, 10), (3, 11), (1, 12), (3, 13), (0, 3), (0, 4), (2, 4), (0, 1), (4, 5)]
# 6.196152422706635
 
#4 points in a square (3D)
#steinerTree(((0,1,0),(1,0,0),(0,0,0),(1,1,0)))
 
#4 points arranged in a tetrahedron (3D)
#steinerTree(((1,0,-1/2**.5),(-1,0,-1/2**.5),(0,1,1/2**.5),(0,-1,1/2**.5)))
 
#8 points arranged in an octahedron (3D)
#steinerTree(((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)))
 
#12 points arranged in an icosahedron (3D) -- 12 points in 3D is pretty much too hard to solve for this code
# phi = (1+5**.5)/2
# steinerTree(((0,1,phi),(0,-1,phi),(0,1,-phi),(0,-1,-phi),\
#              (1,phi,0),(-1,phi,0),(1,-phi,0),(-1,-phi,0),\
#              (phi,0,1),(-phi,0,1),(phi,0,-1),(-phi,0,-1)))
 
#6 points arranged in a right triangular prism (3D); record length 4.501053308. global?
#steinerTree(((0,0,0),(0,1,0),(1,0,0),(0,0,1),(0,1,1),(1,0,1)))
 
#6 points arranged in a equilateral triangular prism (3D) -- record 8.464101615 = 2*(3**.5) + 5
#thick enough so the global min is not just a degeneration of the honeycomb hexagonal tree
#steinerTree(((-1, 0, 0), (1, 0, 0), (0, (3 ** .5), 0), (-1, 0, 2), (1, 0, 2), (0, (3 ** .5), 2)))
 
#6 points in an uneven triangular prism (3D) -- record 7.608
# steinerTree((((1, 0,0),\
# (0.5, 0.8660254037844386,2),\
# (-0.5, 0.8660254037844387,0),\
# (-1,0,2),\
# (-0.5,-0.8660254037844384,0),\
# (0.5,-0.8660254037844386,2))))
 
#8 points in a parallelepiped -- record 14.21773754
#steinerTree(((0,0,0),(1,2,0),(3,0,0),(4,2,0),(1,1,2),(2,3,2),(4,1,2),(5,3,2)))