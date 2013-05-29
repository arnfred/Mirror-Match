"""
Python module for matching two images using proposed algorithm. This is an improved version of louvainmatch

Jonas Toft Arnfred, 2013-05-22
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
import scoring
from itertools import combinations


dist_fun_map = {
	"SIFT"   : weightMatrix.angleDist,
	"SURF"   : weightMatrix.angleDist,
	"ORB"    : weightMatrix.hammingDist,
	"BRISK"  : weightMatrix.hammingDist,
	"BRIEF"  : weightMatrix.hammingDist,
	"FREAK"  : weightMatrix.hammingDist
}

####################################
#                                  #
#           Functions              #
#                                  #
####################################


def match(paths, tresholds, options = {}) : 
	
	# Get parameters
	prune_fun = options.get("prune_fun", weightMatrix.pruneTreshold)
	prune_limit = options.get("prune_limit", 2.5)
	keypoint_type = options.get("keypoint_type", "SIFT")
	descriptor_type = options.get("descriptor_type", "SIFT")
	verbose = options.get("verbose", False)
	split_limit = options.get("split_limit", 50)
	cluster_prune_limit = options.get("cluster_prune_limit", 1.5)

	# Get all feature points
	indices, ks, ds = features.getFeatures(paths, keypoint_type = keypoint_type, descriptor_type = descriptor_type)

	# Calculate weight matrix
	weights = weightMatrix.init(ds, descriptor_type)

	# Get cluster weights
	cluster_weights = prune_fun(weights, prune_limit)

	# Cluster graph
	partitions = cluster(cluster_weights, indices, split_limit = split_limit, prune_limit = cluster_prune_limit, verbose=verbose)
	if verbose : print("%i partitions" % len(set(partitions)))

	# For each treshold, get partition matches
	p = lambda t : getPartitionMatches(partitions, cluster_weights, weights, indices, t)
	match_set = [p(t) for t in tresholds]

	# Get positions
	get_pos = lambda (m_i, m_j) : getMatchPosition(m_i, m_j, ks)
	match_pos_set = [map(get_pos, ms) for ms in match_set]
	
	return match_pos_set



def getMatches(m, ind, treshold) :

	def getMaxMatch(m) :
		for i,row in enumerate(m) :
			s = numpy.argsort(row)
			u = row[s[-2]] / row[s[-1]]
			if u < treshold : yield (i,s[-1])
	
	# Get best matches for rows and columns
	matches = list(getMaxMatch(m))
	
	# Check that matches are both ways
	matches = [(i,m) for i,m in matches if (m,i) in matches and ind[i] == 0 and ind[m] != 0]
	return matches



def cluster(weights, indices, split_limit = 10, prune_limit = 3, verbose = False, rec_level = 0) :
	partitions = louvain.cluster(weights, verbose=verbose)
	if rec_level > 10 : return partitions
	p_set = set(partitions)
	
	r = numpy.arange(0, weights.shape[0])
	for p in p_set :
		partition_mask = partitions == p
		for i,j in combinations(set(indices),2) :
			# Set up masks
			row_mask = partition_mask & (indices == i)
			col_mask = partition_mask & (indices == j)
			index_row = r[row_mask]
			index_col = r[col_mask]
	
			# Get weights
			pij_edges = weights[row_mask][:,col_mask]
			p_edges = weights[partition_mask][:,partition_mask]
			nb_inter_edges = numpy.sum(pij_edges > 0)
			if (numpy.sum(partition_mask) > split_limit) :
				# Prune weights
				p_edges_pruned = weightMatrix.pruneTreshold(p_edges, prune_limit)

				# If there are no edges left, then skip
				if numpy.sum(p_edges_pruned) == 0 : continue

				# normalizing weights
				p_max = numpy.max(p_edges_pruned)
				p_min = numpy.min(p_edges_pruned[p_edges_pruned.nonzero()])
				p_zero = p_edges_pruned == 0
				p_edges_norm = (p_edges_pruned - p_min) / (p_max - p_min)
				p_edges_norm[p_zero] = 0
				
				# cluster
				p_partition = cluster(p_edges_norm, indices[partition_mask], split_limit, prune_limit, verbose, rec_level + 1)
			
				# Update partitioning
				partitions[partition_mask] = p_partition + numpy.max(partitions) + 1
	return partitions


def getPartitionMatches(partitions, weights, full_weights, indices, treshold, verbose = False, ks = None, homography = None) :

	# index
	index = numpy.arange(0, weights.shape[0])
	# Get numpy array of indices
	for p in set(partitions) :

		partition_mask = partitions == p
		partition_weights = weights[partition_mask][:, partition_mask]
		for i,j in combinations(set(indices),2) :

			# Set up masks
			mask_row = partition_mask & (indices == i)
			mask_col = partition_mask & (indices == j)
			mask_both = mask_row | mask_col
			index_row = index[mask_row]
			index_col = index[mask_col]
			index_both = index[mask_row | mask_col]

			# Get weights
			pij_edges = weights[mask_row][:,mask_col]
			pij_edges_both = weights[mask_both][:,mask_both]
			nb_e = numpy.sum(pij_edges > 0)

			# If the cluster as only one edge going from one image to the other
			if nb_e == 1 :
				# Get weight
				(m_i, m_j) = numpy.unravel_index(pij_edges.argmax(), pij_edges.shape)
				w = pij_edges[m_i, m_j]
				
				# Get second largest row and col weight
				sort_row = numpy.sort(full_weights[index_row[m_i],:])
				sort_col = numpy.sort(full_weights[:,index_col[m_j]])
				
				# Test if the maximum match is the both for row and col
				bothways_p = sort_row[-1] == sort_col[-1]
				ratio_row = sort_row[-2] / w
				ratio_col = sort_col[-2] / w

				if bothways_p and (ratio_row < treshold and ratio_col < treshold) :
					(p_i, p_j) = (index_row[m_i], index_col[m_j])
					if verbose : 
						distance = matchDistance(ks[p_i].pt, ks[p_j].pt, homography)
						print("%4i\tEdges: %i\tDistance: %.2f" % (p, nb_e, distance))
					yield (p_i, p_j)

			# If there are several edges
			elif nb_e >= 2 :

				# Collect matches and check if they are beyond treshold
				matches = getMatches(pij_edges_both, indices[mask_both], treshold)
				for m_i,m in matches :
					(p_i, p_j) = (index_both[m_i], index_both[m])
					if verbose :
						distance = matchDistance(ks[p_i].pt, ks[p_j].pt, homography)
						print("%4i\tEdges: %i\tDistance: %.2f" % (p, nb_e, distance))
					yield (p_i, p_j)


def getCoherence(partition_weights, partition_mask, indices, i, j) :
	# Get coherence
	im_masks_i = indices[partition_mask] == i
	im_masks_j = indices[partition_mask] == j
	return -1 * (modularity(partition_weights, im_masks_i) + modularity(partition_weights, im_masks_j))



def getUniqueness(row) :
	nz = row[row.nonzero()]
	max_ind = nz.argsort()
	if (max_ind.size > 1) : return nz[max_ind[-2]] / nz[max_ind[-1]]
	else : return 1.0



def getMatchPosition(i,j, keypoints) :
	pos = features.getPositions([keypoints[i], keypoints[j]])
	return (pos[0], pos[1])



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



def modularity(weights, mask) :
	""" Calculates the modularity of the partition masked by 'mask' in the weight matrix
		Input: Weights [numpy.darray] The weightmatrix used
			   mask [boolean numpy.darray] A mask marking the partition of vertices 
	"""
	K = weights.sum()
	if K == 0 : return 0
	indices = numpy.arange(0,weights.shape[0])
	internal_sum = weights[mask][:, mask].sum()
	external_sum = weights[mask].sum()
	fraction = (internal_sum / (K))
	E_fraction = (external_sum / (K)) ** 2
	return fraction - E_fraction



def matchDistance(p1, p2, hom) :
	""" Given a homography matrix and two points this function calculates
	    the geometric distance between the first point transformed by the
	    homography matrix and the second point
	"""
	m1to2 = hom.dot(numpy.array([p1[0], p1[1], 1]))
	m2to1 = numpy.linalg.inv(hom).dot(numpy.array([p2[0], p2[1], 1]))
	p2_n = m1to2[0:2]/m1to2[2]
	p1_n = m2to1[0:2]/m2to1[2]
	return numpy.linalg.norm(p2_n - p2) + numpy.linalg.norm(p1_n - p1)
