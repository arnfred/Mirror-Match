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



def cluster(graph, nb_clusters=20, iter=100, gamma=1.0, corr="erdos", verbose=False) :

	# Get weights (throws an error if they don't exist)
	weights = graph.edge_properties["weights"]

	clusters = gt.community_structure(graph, iter, nb_clusters, weight=weights, gamma=gamma, corr=corr, verbose=verbose)
	t = [(k, len(list(g))) for k,g in groupby(sorted(clusters.fa))]
	return clusters, len(t)



def getDescriptors(paths, size=32) :
	""" Given a list of paths to images, the function returns a list of 
	    descriptors and keypoints
		Input: paths [list of strings] The paths to the images we are using
		Out:   [pair of list of descriptors and list of keypoints]
	"""
	# Get all images and labels
	labels = map(features.getLabel, paths)
	images = map(features.loadImage, paths)

	# Get feature descriptors
	#keypoints = [features.getKeypoints(feature_keypoint, im) for im in images]
	keypoints = [features.getORBKeypoints(im, size) for im in images]
	data = [features.getDescriptors(feature_descriptor, im, k) for (im, k) in zip(images, keypoints)]
	keypoints, descriptors = zip(*data)

	# Check that we could get descriptors for all images
	if sum(map(lambda d : d == None, descriptors)) > 0 : return (None, None, None)
	indices = [l for i,n in zip(range(len(labels)), map(len, descriptors)) for l in [i]*n]
	return (indices, numpy.concatenate(keypoints), numpy.concatenate(descriptors))



def setPositions(graph, keypoints) :

	positions = display.getPositions(keypoints)

	posMap = graph.new_vertex_property("vector<float>")
	for v in graph.vertices() : posMap[v] = positions[graph.vertex_index[v]]

	# Save weights as a graph property
	graph.vertex_properties["positions"] = posMap

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
