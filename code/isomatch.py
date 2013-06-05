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
import display
import pylab



####################################
#                                  #
#           Functions              #
#                                  #
####################################


def match(paths, thresholds, options = {}) :

	# Get matches in usual format
	def matchFromIndex(i,j) :
		return (features.getPosition(ks[indices == 0][i]), features.getPosition(ks[indices == 1][j]))

	# Get options
	k_init				= options.get("k_init", 50)
	max_iterations		= options.get("max_iterations", 20)
	min_partition_size	= options.get("min_partitions_size", 10)
	max_sd				= options.get("max_sd", 40)
	min_distance		= options.get("min_distance", 25)
	verbose				= options.get("verbose", False)
	keypoint_type		= options.get("keypoint_type", "SIFT")
	descriptor_type		= options.get("descriptor_type", "SIFT")

	# Get images
	images = map(features.loadImage, paths)

	# Get all feature points
	indices, ks, ds = features.getFeatures(paths, keypoint_type = keypoint_type, descriptor_type = descriptor_type)

	# Get positions
	positions = numpy.array(features.getPositions(ks))

	# Get matches
	match_points = getMatchPoints(indices, ks, ds, descriptor_type = descriptor_type)

	if len(match_points) == 0 : return [[] for _ in thresholds]

	# Partition with isodata
	part_1 = isodata.cluster(positions[indices==0], k_init=k_init, max_iterations=max_iterations, min_partition_size=min_partition_size, max_sd=max_sd, min_distance=min_distance)
	part_2 = isodata.cluster(positions[indices==1], k_init=k_init, max_iterations=max_iterations, min_partition_size=min_partition_size, max_sd=max_sd, min_distance=min_distance)

	# Show the clusters
	if verbose : showPartitions(part_1, part_2, indices, images, positions)

	# Get a matrix of the matches so that part_corr_{i,j} is equal to the
	# amount of matches between partition i and j
	part_corr = getLinkMat(part_1, part_2, match_points)

	# For each partition figure out which partitions correspond
	partition_links = [getPartitionLinks(row) for row in part_corr]

	# Get all keypoint matches from the matching clusters
	matches = []
	for i,ms in enumerate(partition_links) :
		for (j,s) in ms :
			matches.extend(getPartitionMatches(match_points, part_1 == i, part_2 == j))
	
	# Filter matches by uniqueness
	p = lambda t : pruneMatches([matchFromIndex(i,j) for ((i,j),u) in matches if u < t])
	match_set = [p(t) for t in thresholds]

	return match_set



def pruneMatches(matches) :
	# filter matches that are deviant from average angle
	def getLength(p1, p2) :
		v = numpy.array([p2[0] - p1[0], p2[1] - p1[1]])
		return numpy.linalg.norm(v)

	def getAngle(p1, p2) :
		# The + 1 is to avoid division with zero when positions overlap
		x_dist = 500
		return numpy.arccos((p2[0] - (p1[0] + x_dist)) / (getLength([p1[0] + x_dist, p1[1]],p2) + 0.001))

	def isAcceptable(l, a) :
		sdv = 1
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


def getLinkMat(part_1, part_2, match_points) :
	""" Get a matrix of the matches so that part_corr_{i,j} is equal to the
	    amount of matches between partition i and j
	"""
	n = len(set(part_1))
	m = len(set(part_2))
	part_corr = numpy.zeros((n, m))
	for p_1 in set(part_1) :
		for p_2 in set(part_2) :
			part_corr[p_1,p_2] = linkCount(match_points, part_1 == p_1, part_2 == p_2)

	return part_corr


def showPartitions(part_1, part_2, indices, images, positions) :
	pylab.figure(frameon=False, figsize=(14,5))
	pylab.subplot(1,2,1)
	display.showPartitions(positions[indices==0], part_1, image = images[0])
	pylab.subplot(1,2,2)
	display.showPartitions(positions[indices==1], part_2, image = images[1])
	pylab.show()


def getMatchPoints(indices, ks, ds, descriptor_type = "SIFT") :

	# Use cv2's matcher to get matching feature points
	bfMatches = features.bfMatch(descriptor_type, ds[indices == 0], ds[indices == 1])

	# Keep relevant data
	match_points = [((j, i), u) for j,(i,s,u) in enumerate(zip(*bfMatches)) if i != None]

	return match_points


# Get matches that pertain to part_1 in image_1 and part_2 in image_2
def getPartitionMatches(match_points, part_1_mask, part_2_mask) :
	return [((i,j),u) for ((i,j), u) in match_points if part_1_mask[i] and part_2_mask[j]]


# Count matches that pertain to part_1 in image_1 and part_2 in image_2
def linkCount(match_points, part_1_mask, part_2_mask) : 
	return len(getPartitionMatches(match_points, part_1_mask, part_2_mask))


# For each partition figure out which partitions correspond
def getPartitionLinks(row) :
	max_links = numpy.max(row)
	pm = [(p_2, ls) for p_2,ls in enumerate(row) if (ls/(float(max_links)) > 0.5 and ls > 5)]
	return pm

