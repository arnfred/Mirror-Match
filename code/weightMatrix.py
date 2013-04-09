
"""
Python module for use with OpenCV to cluster descriptors from several images.

Jonas Toft Arnfred, 2013-03-19
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import numpy
from itertools import dropwhile



####################################
#                                  #
#           Functions              #
#                                  #
####################################

def init(descriptors) :

	# Get all hamming distances based on the descriptors
	distances = hammingDist(descriptors)

	# Normalize distances
	max_d = numpy.max(distances)
	distances_normalized = numpy.array(distances, dtype=numpy.float) / numpy.array(max_d, dtype=numpy.float)
	weights = 1 - distances_normalized

	# Set the self-distances to 0
	weights[(weights == 1)] = 0

	return weights



def get_treshold(weights, edges_per_vertex) :
	n = 300
	nb_vertices = weights.shape[0]
	tresholds = numpy.linspace(1,0.3,n)
	w = [numpy.sum(weights > t) / (2.0*nb_vertices) for t in tresholds]
	q = dropwhile(lambda e : e < edges_per_vertex, w)
	index = len(list(q))
	if (index <= 0) : return 0.3
	if (index >= n) : return 1
	return tresholds[n-index]



def pruneHighest(weights, edges_per_vertex) :
	weights_triu = numpy.triu(weights)
	new_weights = numpy.zeros(weights.shape)
	for i,row in enumerate(weights_triu) :
		indices = row.argsort()[(-1*edges_per_vertex):]
		new_weights[i,indices] = row[indices]
	new_weights = new_weights + new_weights.T
	return new_weights



def pruneTreshold(weights, edges_per_vertex) :
	""" Removes all edges under a certain treshold
	"""
	# Get treshold
	treshold = get_treshold(weights, edges_per_vertex)

	# Update weight matrix
	index = (weights <= treshold) | (weights == 1)
	pruned_weights = weights.copy()
	pruned_weights[index] = 0

	return pruned_weights



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



def resultMatrix(paths, result) :

	def resultDict(result) :
		d = { (p1,p2) : s for _,s,(p1,p2) in result }
		d.update({ (p1,p2) : s for _,s,(p2,p1) in result })
		return d

	d = resultDict(result)
	m = numpy.zeros([len(paths), len(paths)])
	for i,p1 in enumerate(paths) :
		for j,p2 in enumerate(paths) :
			if (p1 != p2) :
				m[i][j] = d[(p1,p2)]

	zeros = (m == -0.1) | (m == 0)
	max_val = numpy.max(m)
	min_val = numpy.min(m)
	ret = (m - min_val) / (max_val - min_val)
	ret[zeros] = 0
	return ret
