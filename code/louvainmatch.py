"""
Python module for matching two images using Isodata clustering of feature
points by geometry

Jonas Toft Arnfred, 2013-04-25
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import numpy
import isodata
import features
import louvain
import weightMatrix
from itertools import combinations



####################################
#                                  #
#           Functions              #
#                                  #
####################################


def match(paths, options = {}) : 
	
	prune_fun = options.get("prune_fun", weightMatrix.pruneTreshold)
	prune_limit = options.get("prune_limit", 5)
	min_edges = options.get("min_edges", 2)
	weight_limit = options.get("weight_limit", 0.7)
	bipartite = options.get("bipartite", False)

	# Get all feature points
	indices, ks, ds = features.getFeatures(paths)

	# Calculate weight matrix (hamming distances)
	full_weights = weightMatrix.init(ds)

	# Get geometric weights
	if bipartite : weights = getGeom(full_weights, ks, indices, weight_limit)
	else : weights = full_weights

	# Get cluster weights
	cluster_weights = prune_fun(weights, prune_limit)

	# Cluster graph
	partitions = louvain.cluster(cluster_weights, verbose=False)

	# Get matches
	matches = [(m1,m2) for (m1,m2,indices) in getPartitionMatches(partitions, cluster_weights, indices, ks, min_edges)]

	return matches


def getPartitionMatches(partitions, weights, indices, keypoints, min_edges = 1, max_variance = 10) :
	
	# Get the edges belong to partition p and image i
	def getEdges(p,i) : 
		return weights[p == partitions & ind == i]
	
	# Get numpy array of indices
	ind = numpy.array(indices)
	for p in set(partitions) :
		for i,j in combinations(set(indices),2) :
			pij_edges = numpy.zeros(weights.shape)
			pij_edges[(p == partitions) & (ind == i)] = weights[(p == partitions) & (ind == i)]
			pij_edges[:, (ind != j)] = 0
			# Check if there are any edges leading to image j
			if numpy.sum(pij_edges) >= min_edges :
				# get the index of the weight between image i and j which is highest
				m_i,m_j = numpy.unravel_index(pij_edges.argmax(), pij_edges.shape)
				# Get the keypoints belonging to this index
				pos = features.getPositions([keypoints[m_i], keypoints[m_j]])
				yield (pos[0], pos[1], (i,j))


def getDistMat(keypoints) :
	# Get positions
	x_list,y_list = zip(*map(features.getPosition, keypoints))
	x = numpy.array(x_list)
	y = numpy.array(y_list)

	# Calculate distances
	x_outer = numpy.outer(x,x)
	y_outer = numpy.outer(y,y)
	x_sq = numpy.outer(x*x,numpy.ones(x.shape))
	y_sq = numpy.outer(y*y,numpy.ones(y.shape))
	#dist_mat = numpy.sqrt(x_sq + x_sq.T + y_sq + y_sq.T - 2*(x_outer + y_outer))
	dist_mat = (x_sq + x_sq.T + y_sq + y_sq.T - 2*(x_outer + y_outer))

	return dist_mat



def getGeom(full_weights, keypoints, indices, limit = 0.7) :

	# Get distance matrix
	dist_mat = getDistMat(keypoints)

	# Normalize
	min_d = numpy.min(dist_mat)
	max_d = numpy.max(dist_mat)
	dist = (dist_mat-min_d) / (max_d - min_d)

	# Inverse and cap
	dist_reversed = (1 - dist)*limit
	dist_reversed[dist==0] = 0.0

	# Get image masks
	im0_mask = indices == 0
	im1_mask = indices == 1

	# Fill in geom with distances and weights
	geom = numpy.zeros(full_weights.shape)
	for i,row in enumerate(geom) :
		row[im0_mask] = dist[i][im0_mask]*im0_mask[i] + full_weights[i][im0_mask]*im1_mask[i]
		row[im1_mask] = dist[i][im1_mask]*im1_mask[i] + full_weights[i][im1_mask]*im0_mask[i]

	return geom
