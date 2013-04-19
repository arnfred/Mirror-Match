"""
Python module for use with OpenCV to cluster descriptors from several images.

Jonas Toft Arnfred, 2013-03-19
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import cv2
import pylab
import numpy
import display
import features
import weightMatrix
import math
import louvain
from itertools import combinations, groupby, tee, product, combinations_with_replacement, dropwhile
import graph_tool.all as gt



####################################
#                                  #
#           Properties             #
#                                  #
####################################

feature_keypoint = "ORB"
feature_descriptor = "BRIEF"
N = 600


####################################
#                                  #
#           Functions              #
#                                  #
####################################



def clusterPotts(graph, nb_clusters=20, iter=100, gamma=1.0, corr="erdos", verbose=False) :

	# Get weights (throws an error if they don't exist)
	weights = graph.edge_properties["weights"]

	clusters = gt.community_structure(graph, iter, nb_clusters, weight=weights, gamma=gamma, corr=corr, verbose=verbose)
	t = [(k, len(list(g))) for k,g in groupby(sorted(clusters.fa))]
	return clusters, len(t)



def setPositions(graph, keypoints) :

	positions = features.getPositions(keypoints)

	pos_x = graph.new_vertex_property("float")
	pos_y = graph.new_vertex_property("float")
	for v in graph.vertices() : 
		pos_x[v] = positions[graph.vertex_index[v]][0]
		pos_y[v] = positions[graph.vertex_index[v]][1]

	# Save weights as a graph property
	graph.vertex_properties["x"] = pos_x
	graph.vertex_properties["y"] = pos_y

	return graph



def setFeatureIndices(graph, indices) :

	# Create indicemap
	ind = graph.new_vertex_property("int")
	ind.fa = indices
	graph.vertex_properties["indices"] = ind

	# Define class colors
	class_colors = graph.new_vertex_property("vector<float>")
	colors = [[0.1, 0.1, 0.1, 0.9], [0.7, 0.7, 0.7, 0.9], [0.3, 0.3, 0.3, 0.9], [0.9, 0.9, 0.9, 0.9], [0.5, 0.5, 0.5, 0.9],
			  [0.2, 0.2, 0.2, 0.9], [0.8, 0.8, 0.8, 0.9], [0.4, 0.4, 0.4, 0.9], [0.6, 0.6, 0.6, 0.9], [1.0, 1.0, 1.0, 0.9]] 
	for v in graph.vertices() : class_colors[v] = colors[ind[v]]
	graph.vertex_properties["class_colors"] = class_colors



def getImageIndices(paths) : 
    labels = map(features.getLabel, paths)
    i = 0
    d = {}
    for e in labels :
        if e not in d :
            d[e] = i
            i += 1
    return [d[e] for e in labels]



def init(weights) :

	# This is a faster way to initialize a graph. It's not random.
	N = weights.shape[0]
	graph = gt.random_graph(N, lambda: N - 1, directed=False, random=False)

	# Because the random graph is generated with scrambled indices we need to reorder
	graph.reindex_edges()

	# Set weights
	weighted_graph = setWeights(graph, weights)

	return weighted_graph



def setWeights(graph, weights) :

	# Set weights
	weight_prop = graph.new_edge_property("float")
	weight_prop.fa = [(weights[graph.vertex_index[e.target()], 
							 graph.vertex_index[e.source()]]) for e in graph.edges()]

	# Create mask to filter edges with 0 weight
	non_zero = graph.new_edge_property("bool")
	non_zero.fa = weight_prop.fa > 0

	# Create new graph with fewer edges
	new_graph = gt.GraphView(graph, efilt=non_zero)

	# Store weights as graph property and set degree
	new_graph.edge_properties["weights"] = weight_prop
	setDegree(new_graph, weights)

	return new_graph



def setDegree(graph, weights) :
	degrees = numpy.array(map(numpy.sum, weights))
	g_degrees = graph.new_vertex_property("float")
	g_degrees.fa = degrees
	graph.vp["degrees"] = g_degrees



def getFilter(graph, partitioning, indices) :
	""" Given a partitioning and some indices, return a boolean vertex property
	    marking the vertices as true if they belong to the partitions
	    input: graph [Graph]
	           partitioning: [Integer Vertex property] A partitioning of the graph
	           indices: An integer or a list of integers denoting which clusters to use
	    output: Boolean vertex property
	"""
	if isinstance(indices, ( int, long ) ) : indices = [indices]
	f_new = numpy.zeros(partitioning.fa.shape, dtype=numpy.bool)
	for i in indices :
		f_new = (partitioning.fa == i) | f_new

	# Create a filter
	f = graph.new_vertex_property("bool")
	f.fa = f_new
	return f



def andFilter(graph, f, partitioning, indices) :
	""" Given a filter, a partitioning and some indices, return a boolean vertex property
	    marking the vertices as true if they belong to the partitions and to the previous filter
	    input: graph [Graph]
	           f [Boolean vertex property] The filter we are extending
	           partitioning: [Integer Vertex property] A partitioning of the graph
	           indices: An integer or a list of integers denoting which clusters to use
	    output: Boolean vertex property
	"""
	if isinstance(indices, ( int, long ) ) : indices = [indices]
	f_new = numpy.zeros(partitioning.fa.shape, dtype=numpy.bool)
	for i in indices :
		f_new = (partitioning.fa == i) | f_new

	# Create a filter
	f_ret = graph.new_vertex_property("bool")
	f_ret.fa = f_new & numpy.array(f.fa, dtype=numpy.bool)
	return f_ret



def getPartition(graph, partitioning, partition_indices) :
	""" Returns a partition corresponding to the partitioning and the indices
	"""
	# Make sure indices are a list
	if isinstance(partition_indices, ( int, long ) ) : partition_indices = [partition_indices]

	# Add all qualified vertices to filter
	f = numpy.zeros(partitioning.fa.shape, dtype=numpy.bool)
	for i in partition_indices :
		f = (partitioning.fa == i) | f
		
	# Create new filter
	new_f = graph.new_vertex_property("bool")
	new_f.fa = f

	# Create graph_view for the cluster
	return gt.GraphView(graph, vfilt=new_f)



def setPartition(graph, partition_array) :
	p = graph.new_vertex_property("int")
	p.fa = partition_array
	graph.vp["partitions"] = p
	return p



def makeFaceGraph(paths, result, treshold=3, lower_is_better=False) :
	initWeights = weightMatrix.resultMatrix(paths, result)
	if lower_is_better : initWeights = 1 - initWeights
	weights = weightMatrix.pruneTreshold(initWeights, treshold)
	graph = init(weights)
	indices = getImageIndices(paths)
	setFeatureIndices(graph, indices)
	path_prop = graph.new_vertex_property("string")
	for v, p in zip(graph.vertices(), paths) : path_prop[v] = p
	path_prop.fa = paths
	graph.vp["paths"] = path_prop
	return graph



def getMeanPos(graph) :
	x_prop = graph.vp["x"]
	y_prop = graph.vp["y"]
	pos = [(x_prop[v], y_prop[v]) for v in graph.vertices()]
	(xs, ys) = zip(*pos)
	return (numpy.mean(xs), numpy.mean(ys))

def getVarPos(graph) :
	x_prop = graph.vp["x"]
	y_prop = graph.vp["y"]
	pos = [(x_prop[v], y_prop[v]) for v in graph.vertices()]
	(xs, ys) = zip(*pos)
	return numpy.sqrt((numpy.var(xs) + numpy.var(ys)) / 2.0) * 0.1
	
def make_trait_graph(graph, partitions, partition_indices, scores) :
	index_prop = graph.vp["indices"]
	assert(len(partition_indices) == len(scores))
	
	# Initialize trait graph
	trait_graph = gt.Graph(directed=False)
	trait_partition = trait_graph.new_vertex_property("int")
	trait_weight = trait_graph.new_edge_property("float")
	trait_x = trait_graph.new_vertex_property("float")
	trait_y = trait_graph.new_vertex_property("float")
	trait_variance = trait_graph.new_vertex_property("float")
	trait_indices = trait_graph.new_vertex_property("int")
	trait_class_colors = trait_graph.new_vertex_property("vector<float>")
	trait_edge_colors = trait_graph.new_edge_property("vector<float>")
	colors = [[0.1, 0.1, 0.1, 0.9], [0.7, 0.7, 0.7, 0.9]]
	edge_colors = [[0.5, 0.0, 0.0, 0.9], [0.0, 0.4, 0.0, 0.9]]
	
	# Fill graph with vertices and edges
	for s, c in zip(scores, partition_indices) :
		f_c = getFilter(graph, partitions, c)
		indices = [0,1]
		fs = [andFilter(graph, f_c, index_prop, i) for i in indices]
		gs = [gt.GraphView(graph, vfilt=f) for f in fs]
		vs = [trait_graph.add_vertex() for _ in gs]
		e = trait_graph.add_edge(vs[0], vs[1])
		if (s < 0) :
			trait_weight[e] = s*(-200)
			trait_edge_colors[e] = edge_colors[0]
		else :
			trait_weight[e] = s*200
			trait_edge_colors[e] = edge_colors[1]
		for v,g,i in zip(vs,gs, indices) : 
			(x,y) = getMeanPos(g)
			trait_x[v] = x
			trait_y[v] = y
			trait_variance[v] = getVarPos(g)
			trait_partition[v] = c
			trait_indices[v] = i
			trait_class_colors[v] = colors[i]
	
	# Return finished graph
	trait_graph.vp["partitions"] = trait_partition
	trait_graph.ep["weights"] = trait_weight
	trait_graph.vp["x"] = trait_x
	trait_graph.vp["y"] = trait_y
	trait_graph.vp["variance"] = trait_variance
	trait_graph.vp["indices"] = trait_indices
	trait_graph.vp["class_colors"] = trait_class_colors
	trait_graph.ep["edge_colors"] = trait_edge_colors
	return trait_graph
