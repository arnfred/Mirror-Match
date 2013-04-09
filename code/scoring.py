"""
Python module for scoring the graphs made of feature points of two images

Jonas Toft Arnfred, 2013-04-05
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import louvain
import weightMatrix
import features
import graph_tool.all as gt
import numpy
import math

####################################
#                                  #
#           Functions              #
#                                  #
####################################


def scoreImages(paths, cluster_edges = 3, score_edges = 20, scoring = lambda m,c : m + c if c > 0 else 0.0, default_val = 0.0) :
	""" Given paths to two images, the images are scored based on how 
	    well their traits match
	"""
	# Get features
	indices, keypoints, descriptors = features.getFeatures(paths, size=32)
	if descriptors == None :
		print(paths)
		print("No descriptors found. Returning score %i" % default_val)
		return default_val

	# Get weights
	full_weights = weightMatrix.init(descriptors)
	score_weights = weightMatrix.pruneTreshold(full_weights, score_edges)
	cluster_weights = weightMatrix.pruneHighest(score_weights, cluster_edges)

	# Cluster graph
	partitions = louvain.cluster(cluster_weights)

	# Match the traits
	scores, cluster_indices = scoreWeights(score_weights, partitions, indices, scoring=scoring)

	# Get labels
	[l1, l2] = [features.getLabel(p) for p in paths]

	# Get score
	if len(scores) > 3 :
		score = sum(scores)
		print("Score: %0.4f for %s and %s" % (score,l1,l2))
	else :
		score = default_val
		print("Score: %0.4f for %s and %s (Less than four matches)" % (score,l1,l2))

	return score



def scoreWeights(weights, partitions, image_indices, scoring = lambda m,c : m + c if c > 0 else 0.0, verbose=False) :
	""" This function returns a score given the weights and a partition
	    input: weights [numpy.darray] the weights of the graph
	           partitions [numpy.darray] An array of partition indices
			   image_indices [numpy.darray] A partition designating which image each point belongs to
	           scoring [Int -> Int -> Float] A function that scores a cluster given the modularity and coherence
	           verbose [Boolean = False] Enable extra output
	"""

	def printResult(i, m, c, w) : 
		print("[%i]" % i + "\tm: %+0.4f" % m + ", c: %+0.4f" % c + ", u_s: %+0.4f" % scoring(m,c) + ", s: %+0.4f" % (scoring(m,c) * w) + ", w: %0.2f" % w)
	
	def in_range(partition_mask, image_masks) :
		return (image_masks[1].sum() != 0 and image_masks[0].sum() != 0)
	
	def score(index, im_masks, partition_mask, weight) :
		
		partition_weights = weights[partition_mask][:, partition_mask]
		m = modularity(weights, partition_mask)
		c = -1 * (modularity(partition_weights, im_masks[1]) + modularity(partition_weights, im_masks[0]))
		if verbose : printResult(index, m,c,weight)
		return scoring(m,c) * weight

	# Make sure indices is array
	ind = numpy.array(image_indices)
	
	# Define range and precalculate modularity
	partition_set = set(partitions)
	
	# Define masks
	p_masks = [(p, partitions == p) for p in partition_set]
	i_masks = [(ind[p] == 0, ind[p] == 1) for (i,p) in p_masks]
	
	# Create list of data
	data = [(p_index, modularity(weights, p), p, im) for (p_index,p),im in zip(p_masks, i_masks) if in_range(p,im)]
	mods_sum = numpy.sum([m for i, m, p, im in data])
	data_norm = [(i, m/mods_sum, p, im) for i, m, p, im in data]
	scores = [score(i, im, p, m) for i, m, p, im in data_norm]
	partition_indices = [i for i, m, p, im in data]
	
	return scores, partition_indices



def modularity(weights, mask) :
	""" Calculates the modularity of the partition masked by 'mask' in the weight matrix
		Input: Weights [numpy.darray] The weightmatrix used
			   mask [boolean numpy.darray] A mask marking the partition of vertices 
	"""
	K = weights.sum()
	indices = numpy.arange(0,weights.shape[0])
	internal_sum = weights[mask][:, mask].sum()
	external_sum = weights[mask].sum()
	fraction = (internal_sum / (K))
	E_fraction = (external_sum / (K)) ** 2
	return fraction - E_fraction


