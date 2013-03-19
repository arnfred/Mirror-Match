"""
Python module for use with openCV to cluster descriptors from several images.

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
from itertools import combinations
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



def getDescriptors(paths) :
	""" Given a list of paths to images, the function returns a list of 
	    descriptors and keypoints
		Input: paths [list of strings] The paths to the images we are using
		Out:   [pair of list of descriptors and list of keypoints]
	"""
	# Get all images and labels
	labels = map(features.getLabel, paths)
	images = map(features.loadImage, paths)

	# Get feature descriptors
	keypoints = [features.getKeypoints(feature_keypoint, im) for im in images]
	data = [features.getDescriptors(feature_descriptor, im, k) for (im, k) in zip(images, keypoints)]
	keypoints, descriptors = zip(*data)
	indices = [l for i,n in zip(range(len(labels)), map(len, data)) for l in [i]*n]
	return (indices, numpy.concatenate(keypoints), numpy.concatenate(descriptors))



def initGraph() :
	""" To speed up the calculations I reuse the same graph over and over
	"""
	# Initialize graph
	graph = gt.Graph()

	# Add N vertices
	vertices = [graph.add_vertex() for _ in range(N)]

	# Add an edge between every vertex
	edges = [graph.add_edge(v1,v2) for (v1,v2) in combinations(graph.vertices(),2)]

	return graph



def trimGraph(descriptors, graph) :

	# First filter the edges and vertices to fit with the amount of descriptors
	dv = graph.new_vertex_property("bool")
	n = len(descriptors)
	dv.a = [True]*n + [False]*(N - n)

	# Filter edges
	de = graph.new_edge_property("bool")
	de.a = [dv[e.source()] and dv[e.target()] for e in graph.edges()]

	return gt.GraphView(graph, vfilt=dv, efilt=de, directed=False)



def setWeights(descriptors, graph) :
	dists = hammingDist(descriptors)
	weights = graph.new_edge_property("int")
	weights.fa = [(dists[graph.vertex_index[e.target()], graph.vertex_index[e.source()]] + 1) for e in graph.edges()]
	return weights





def hammingDist(descriptors) :
	""" Returns a matrix of size n x n where n is the amount of descriptors
		output[0][1] is equal to the hamming distance between descriptors[0] and descriptors[1]
		where output is the matrix returned from this function
	"""
	# rearrange inputs
	binary = numpy.array(numpy.unpackbits(descriptors), dtype=numpy.bool)
	binary.shape = (descriptors.shape[0], descriptors.shape[1] * 8)

	# Initialize return matrix
	result = numpy.zeros([descriptors.shape[0], descriptors.shape[0]], dtype=numpy.uint8)

	# Fill result matrix and return
	for (i, bin_row) in enumerate(binary) :
		result[i] = numpy.sum(numpy.bitwise_xor(binary, bin_row), 1)
	return result
