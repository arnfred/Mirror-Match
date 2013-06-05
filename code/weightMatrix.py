
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
import math
import features
from itertools import dropwhile



####################################
#                                  #
#           Functions              #
#                                  #
####################################


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



# Compute D1 * D2.T, and find approx dist with arccos
def angleDist(D) : 
	norms = numpy.array([numpy.linalg.norm(row) for row in D])
	D_n = D / norms[:, numpy.newaxis]
	dotted = D_n.dot(D_n.T)
	dotted = dotted / numpy.max(dotted)
	return numpy.arccos(dotted)



def init(descriptors, descriptor_type) :

	dist_fun_map = {
		"SIFT"   : angleDist,
		"SURF"   : angleDist,
		"ORB"    : hammingDist,
		"BRISK"  : hammingDist,
		"BRIEF"  : hammingDist,
		"FREAK"  : hammingDist
	}

	dist_max_map = {
		"SIFT"   : numpy.pi,
		"SURF"   : numpy.pi,
		"ORB"    : 255,
		"BRISK"  : 255,
		"BRIEF"  : 255,
		"FREAK"  : 255
	}
	
	dist_measure = dist_fun_map.get(descriptor_type, hammingDist)
	dist_max = dist_max_map.get(descriptor_type, 255.0)

	# Get all hamming distances based on the descriptors
	distances = dist_measure(descriptors)

	# Normalize distances
	#max_d = float(numpy.max(distances))
	#min_d = float(numpy.min(distances))
	min_d = 0
	max_d = dist_max
	distances_normalized = (distances - min_d) / float(max_d - min_d)
	weights = 1 - distances_normalized

	# Set the self-distances to 0
	numpy.fill_diagonal(weights, 0)

	return weights



def get_treshold(weights, edges_per_vertex, n=600, start=0.0) :
	nb_vertices = weights.shape[0]
	tresholds = numpy.linspace(1,start,n)
	w = [numpy.sum(weights > t) / (2.0*nb_vertices) for t in tresholds]
	q = dropwhile(lambda e : e < edges_per_vertex, w)
	index = len(list(q))
	if (index <= 0) : return start
	if (index >= n) : return 1
	return tresholds[n-index]



def pruneRows(weights, fraction, n = 1000) :
	def getTres() :
		row_max = numpy.max(weights, axis=0)
		tresholds = numpy.linspace(numpy.max(row_max),numpy.min(row_max),n)
		w = [numpy.sum(row_max > t) / float(row_max.size) for t in tresholds]
		q = dropwhile(lambda e : e < fraction, w)
		index = len(list(q))
		return tresholds[n-index]
	
	treshold = getTres()

	# Update weight matrix
	index = (weights <= treshold) | (weights == 1)
	pruned_weights = weights.copy()
	pruned_weights[index] = 0

	return pruned_weights


def get_fraction(weights, fraction, n=600, start=0.0) :
	nb_weights = float(weights.size)
	tresholds = numpy.linspace(1,start,n)
	w = [numpy.sum(weights > t) / nb_weights for t in tresholds]
	q = dropwhile(lambda e : e < fraction, w)
	index = len(list(q))
	if (index <= 0) : return start
	if (index >= n) : return 1
	return tresholds[n-index]



def pruneFraction(weights, fraction, n=500, start=0.0) :
	# Get treshold
	treshold = get_fraction(weights, fraction, n=n, start=start)

	# Update weight matrix
	index = (weights <= treshold) | (weights == 1)
	pruned_weights = weights.copy()
	pruned_weights[index] = 0

	return pruned_weights



def pruneHighest(weights, edges_per_vertex,n=0,start=0) :
	weights_triu = numpy.triu(weights)
	new_weights = numpy.zeros(weights.shape)
	for i,row in enumerate(weights) :
		indices = row.argsort()[(-1*edges_per_vertex):]
		new_weights[i,indices] = row[indices]
	new_weights[new_weights==0] += new_weights.T[new_weights==0]
	return new_weights



def pruneThreshold(weights, edges_per_vertex, n=600, start=0.0) :
	""" Removes all edges under a certain treshold
	"""
	# Get treshold
	treshold = get_treshold(weights, edges_per_vertex, n=n, start=start)

	# Update weight matrix
	index = (weights <= treshold) | (weights == 1)
	pruned_weights = weights.copy()
	pruned_weights[index] = 0

	return pruned_weights



def resultMatrix(paths, result) :

	def resultDict(result) :
		d = { (p1,p2) : s for _,s,(p1,p2) in result}
		d.update({ (p1,p2) : s for _,s,(p2,p1) in result })
		return d

	d = resultDict(result)
	m = numpy.zeros([len(paths), len(paths)])
	for i,p1 in enumerate(paths) :
		for j,p2 in enumerate(paths) :
			if (p1 != p2) :
				if math.isnan(d[(p1,p2)]) : m[i][j] = 0
				else : m[i][j] = d[(p1,p2)]

	max_val = numpy.max(m)
	min_val = numpy.min(m)
	ret = (m - min_val) / (max_val - min_val)
	return ret
