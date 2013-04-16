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



def scoreWeights_2(weights, partitions, image_indices, scoring = lambda m,c : m + c if c > 0 else 0.0, verbose=False) :

	def printResult(i, bipartite, isolate, w) : 
		print("[%i]" % i + "\tbipartite: %+0.4f" % bipartite + ", isolate: %+0.4f" % isolate + ", ratio: %0.4f" % (isolate/bipartite) + ", weight: %0.2f" % w)
	
	def in_range(partition_mask, image_masks) :
		return (image_masks[1].sum() > 1 and image_masks[0].sum() > 1)

	def score(index, im_masks, partition_mask, weight) :

		partition_weights = weights[partition_mask][:, partition_mask]
		bipartite_weights = partition_weights[im_masks[0]][:,im_masks[1]]
		isolated_weights_im0 = partition_weights[im_masks[0]][:,im_masks[0]]
		isolated_weights_im1 = partition_weights[im_masks[1]][:,im_masks[1]]
		isolated_sum = numpy.sum(isolated_weights_im0) + numpy.sum(isolated_weights_im1)
		isolated_nb = numpy.sum(isolated_weights_im0 != 0.0) + numpy.sum(isolated_weights_im1 != 0.0)
		bipartite_mean = numpy.sum(bipartite_weights) / float(numpy.sum(bipartite_weights != 0.0))
		if (isolated_nb > 0) :
			isolated_mean = (isolated_sum) / float(isolated_nb)
		else :
			isolated_mean = bipartite_mean
		if verbose : printResult(index, bipartite_mean, isolated_mean, weight)
		return bipartite_mean * weight

	# Make sure indices is array
	ind = numpy.array(image_indices)
	
	# Define range and precalculate modularity
	partition_set = set(partitions)
	
	# Define masks
	p_masks = [(p, partitions == p) for p in partition_set]
	i_masks = [(ind[p] == 0, ind[p] == 1) for (i,p) in p_masks]
	
	# Create list of data
	data = [(p_index, p.size, p, im) for (p_index,p),im in zip(p_masks, i_masks) if in_range(p,im)]
	normalize_sum = numpy.sum([n for i, n, p, im in data])
	data_norm = [(i, (1.0*n)/normalize_sum, p, im) for i, n, p, im in data]
	scores = [score(i, im, p, w)/float(10) for i, w, p, im in data_norm]
	partition_indices = [i for i, m, p, im in data]

	return scores, partition_indices



def scoreWeights(weights, partitions, image_indices, scoring = lambda m,c : m + c if c > 0 else 0.0, verbose=False, normalize=True) :
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
		return (image_masks[1].sum() > 0 and image_masks[0].sum() > 0)

	
	def score(index, im_masks, partition_mask, weight) :
		w = weight if normalize else 1.0
		partition_weights = weights[partition_mask][:, partition_mask]
		m = modularity(weights, partition_mask)
		c = -1 * (modularity(partition_weights, im_masks[1]) + modularity(partition_weights, im_masks[0]))
		if verbose : printResult(index, m,c,w)
		return scoring(m,c) * w

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
	scores = [score(i, im, p, w) for i, w, p, im in data_norm]
	partition_indices = [i for i, m, p, im in data]
	
	return scores, partition_indices



def scoreImages(paths, 
		cluster_edges 	= 3, 
		score_edges 	= 40, 
		size 			= 36, 
		withGeometry 	= True, 
		withCertainty 	= True, 
		cluster_prune 	= weightMatrix.pruneHighest,
		score_prune 	= weightMatrix.pruneTreshold,
		normalize		= True,
		score_type		= scoreWeights) :
	""" Given paths to two images, the images are scored based on how 
	    well their traits match
	"""

	default_val = 0.0

	# Get features
	indices, keypoints, descriptors = features.getFeatures(paths, size=size)
	if descriptors == None :
		print(paths)
		print("No descriptors found. Returning score %i" % default_val)
		return default_val

	# Get weights
	full_weights = weightMatrix.init(descriptors)
	score_weights = score_prune(full_weights, score_edges)
	cluster_weights = cluster_prune(score_weights, cluster_edges)

	# Cluster graph
	partitions = louvain.cluster(cluster_weights)

	# Match the traits
	scores, partition_indices = score_type(score_weights, partitions, indices, scoring=lambda m,c : m + c if c > 0 else 0.0, normalize=normalize)

	# Get the geometric multiplier
	geom_multiplier = geometryMultiplier(partitions, partition_indices, numpy.array(indices), keypoints)

	# Get the certainty factor
	certainty_factor = certaintyFactor(len(partition_indices)) if withCertainty else 1.0
	score_sum = sum([s*m for s,m in zip(geom_multiplier, scores)]) if withGeometry else sum(scores)

	# Get final score by multiplying certainty
	final_score = score_sum * certainty_factor

	# Get labels and print
	[l1, l2] = [features.getLabel(p) for p in paths]
	print("Score: %0.4f for %s and %s (clusters: %i)" % (final_score,l1,l2, len(partition_indices)))

	return final_score





def getPartitionDeviation(partition_mask, image_mask, kpts) :
	
	def getSD(pos) :
		ps = [(p[0],p[1]) for p in pos]
		(xs, ys) = zip(*ps)
		return numpy.sqrt((numpy.var(xs) + numpy.var(ys)) / 2.0)
	
	kpt_array = numpy.array(kpts)
	im0_kpts = kpt_array[partition_mask & image_mask[0]]
	im1_kpts = kpt_array[partition_mask & image_mask[1]]
	pos0 = features.getPositions(im0_kpts)
	pos1 = features.getPositions(im1_kpts)
	sd0 = getSD(pos0)
	sd1 = getSD(pos1)
	return sd0,sd1

def geometryMultiplier(partition, partition_indices, image_indices, kpts) :
	sds = [getPartitionDeviation(partition == p, (image_indices == 0, image_indices == 1), kpts) for p in partition_indices]
	multiplier = [1.0 if max(sd) < 3 else 3.0/max(sd) for sd in sds]
	return multiplier



def certaintyFactor(n) : return n*(40.0-n)/400.0 if n<20 else 1.0




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


