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
#           Attributes             #
#                                  #
####################################

feature_keypoint = "ORB"
feature_descriptor = "BRIEF"
graph = gt.Graph()


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



def initGraph(N = 600) :
	""" To speed up the calculations I reuse the same graph over and over
	"""
	# Add N vertices
	vertices = [graph.add_vertix() for _ in range(N)]

	# Add an edge between every vertex
	edges = [g.add_edge(v1,v2) for (v1,v2) in combinations(g.vertices(),2)]



# TODO: time this function
def getGraph(descriptors) :

	# First filter the edges and vertices to fit with the amount of descriptors
	dg = graph.new_edge_property("bool")
	N = graph.num_vertices()
	n = len(descriptors)
	assert (n < N)
	dg.a = [True]*n + [False]*(N - n)

	# Filter edges





def hamming_mat(descriptors) :
	""" Returns a matrix of size n x n where n is the amount of descriptors
		output[0][1] is equal to the hamming distance between descriptors[0] and descriptors[1]
		where output is the matrix returned from this function
	"""
	# rearrange inputs
	binary = array(numpy.unpackbits(rows), dtype=numpy.bool)
	binary.shape = (rows.shape[0], rows.shape[1] * 8)

	# Initialize return matrix
	result = numpy.zeros([rows.shape[0], rows.shape[0]], dtype=numpy.uint8)

	# Fill result matrix and return
	for (i, bin_row) in enumerate(binary) :
		result[i] = numpy.sum(numpy.bitwise_xor(binary, bin_row), 1)
	return result
