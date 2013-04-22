"""
Python module for matching two images using louvain clustering of the
feature points to group traits.

Jonas Toft Arnfred, 2013-04-22
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import numpy
import louvain
import features
import weightMatrix
import display
from itertools import combinations



def testMatch(paths, homography, match_fun, prune_fun = weightMatrix.pruneTreshold, prune_limit = 5, weight_limit = 0.7) :

	# Get matches
	matches = match_fun(paths, prune_fun, prune_limit, weight_limit)

	# Get distances
	dist = [matchDistance(m1,m2,homography) for (m1,m2) in matches]

	# Display result
	print
	display.distHist(dist)

	return matches



def getACRPaths(compare_id = 2, img_type = "graf") :

	# Dictionary to load the files
	file_endings = {
		"bark" : "ppm",
		"bikes" : "ppm",
		"boat" : "pgm",
		"graf" : "ppm",
		"leuven" : "ppm",
		"trees" : "ppm",
		"ubc" : "ppm",
		"wall" : "ppm"
	}

	# Get paths
	img_types = ["bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall"]
	img_ending = file_endings[img_type]
	i1 = 1
	i2 = compare_id
	p1 = "../../images/acr/" + img_type + ("/img%i." % i1) + img_ending
	p2 = "../../images/acr/" + img_type + ("/img%i." % i2) + img_ending
	paths = [p1, p2]

	# Load and parse homography
	hom_path = "../../images/acr/" + img_type + "/H%ito%ip" % (i1,i2)
	hom = numpy.array([map(lambda s : float(s.strip()), line.strip().split()) for i,line in enumerate(open(hom_path)) if i < 3])

	return paths, hom



def clusterMatch(paths, prune_fun = weightMatrix.pruneTreshold, prune_limit = 5, weight_limit = None) :

	# Get all feature points
	indices, ks, ds = features.getFeatures(paths)

	# Calculate weight matrix (hamming distances)
	full_weights = weightMatrix.init(ds)
	cluster_weights = prune_fun(full_weights, prune_limit)

	# Cluster graph
	partitions = louvain.cluster(cluster_weights, verbose=True)

	# Get matches
	matches = [(m1,m2) for (m1,m2,indices) in getPartitionMatches(partitions, cluster_weights, indices, ks)]

	# Prune matches
	matches_pruned = pruneMatches(matches)

	return matches_pruned




def clusterWeightMatch(paths, prune_fun = weightMatrix.pruneTreshold, prune_limit = 5, weight_limit = 0.7) :

	# Get all feature points
	indices, ks, ds = features.getFeatures(paths)

	# Calculate weight matrix (hamming distances)
	full_weights = weightMatrix.init(ds)

	# Get geometric weights
	geom_weights = getGeom(full_weights, ks, indices, weight_limit)

	# Prune weights
	cluster_weights = prune_fun(geom_weights, prune_limit)

	# Cluster graph
	partitions = louvain.cluster(cluster_weights, verbose=True)

	# Get matches
	matches = [(m1,m2) for (m1,m2,indices) in getPartitionMatches(partitions, cluster_weights, indices, ks)]

	# Prune matches
	matches_pruned = pruneMatches(matches)

	return matches_pruned




def standardMatch(paths, prune_fun = None, prune_limit = 100, weight_limit = None) :

	# Get all feature points
	indices, ks, ds = features.getFeatures(paths)

	# Use cv2's matcher to get matching feature points
	bfMatches = features.bfMatch("BRIEF", ds[indices == 0], ds[indices == 1])
	#bfMatches = features.match("BRIEF", ds[ind == 0], ds[ind == 1])

	# Get matches in usual format
	def matchFromIndex(i,j) :
		return (features.getPosition(ks[indices == 0][i]), features.getPosition(ks[indices == 1][j]))

	match_score = [(matchFromIndex(j,i), s, u) for j,(i,s,u) in enumerate(zip(*bfMatches)) if i != None]
	matches, scores, uniques = zip(*match_score)

	# Take only the n best
	top_n = numpy.argsort(scores)[0:prune_limit]
	matches_top = numpy.array(matches)[top_n]

	# Prune matches
	matches_pruned = pruneMatches(matches_top)

	return matches_pruned


def matchDistance(p1, p2, hom) :
	""" Given a homography matrix and two points this function calculates
	    the geometric distance between the first point transformed by the
	    homography matrix and the second point
	"""
	m1to2 = hom.dot(numpy.array([p1[0], p1[1], 1]))
	(x,y) = (m1to2[0]/m1to2[2], m1to2[1]/m1to2[2])
	return numpy.sqrt((x - p2[0])**2 + (y - p2[1])**2)



def getPartitionMatches(partitions, weights, indices, keypoints) :
	
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
			if numpy.sum(pij_edges) > 0 :
				# get the index of the weight between image i and j which is highest
				m_i,m_j = numpy.unravel_index(pij_edges.argmax(), pij_edges.shape)
				# Get the keypoints belonging to this index
				pos = features.getPositions([keypoints[m_i], keypoints[m_j]])
				yield (pos[0], pos[1], (i,j))



def pruneMatches(matches) :
	# filter matches that are deviant from average angle
	def getLength(p1, p2) :
		v = numpy.array([p2[0] - p1[0], p2[1] - p1[1]])
		return numpy.linalg.norm(v)

	def getAngle(p1, p2) :
		# The + 1 is to avoid division with zero when positions overlap
		return numpy.arccos((p2[0] - (p1[0] + 1)) / (getLength([p1[0] + 1, p1[1]],p2) + 0.001))

	def isAcceptable(l, a) :
		sdv = 1.5
		within_length = l < (mdn_length + sdv*sdv_length) and l > (mdn_length - sdv*sdv_length)
		within_angle = a < (mdn_angle + sdv*sdv_angle) and a > (mdn_angle - sdv*sdv_angle)
		return within_length and within_angle

	# Get the length and angle of each match
	lengths = [ getLength(p1,p2) for p1,p2 in matches]
	angles = [ getAngle(p1,p2) for p1,p2 in matches]

	# Calculate a bit of statistics
	mdn_length = numpy.median(lengths)
	sdv_length = numpy.sqrt(numpy.var(lengths))
	mdn_angle = numpy.median(angles)
	sdv_angle = numpy.sqrt(numpy.var(angles))

	return [m for m,l,a in zip(matches, lengths, angles) if isAcceptable(l, a)]



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
