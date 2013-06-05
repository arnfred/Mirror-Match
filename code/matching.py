"""
Python module for matching two images using various techniques including:
- Louvain clustering of feature points by similarity
- Isodata clustering of feature points by geometry
- Nearest neighbor matching of feature points by similarity
- As above but filtering out points that aren't unique

Jonas Toft Arnfred, 2013-04-22
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import numpy
import isomatch
import clustermatch
import isomatch
import mirrormatch
import features
import weightMatrix
import display
import pylab
import fnmatch
import os
import re
import itertools
import random



####################################
#                                  #
#           Functions              #
#                                  #
####################################


def testMatch(paths, homography, match_fun, options = {}, return_matches = False, verbose = False) :

	distinct_threshold = options.get("distinct_threshold", 5)
	thresholds = options["thresholds"]

	# Get matches over thresholds
	match_pos_set = match_fun(paths, thresholds, options)

	# Get distances
	getDist = lambda ms : [matchDistance(m1,m2,homography) for (m1,m2) in ms]
	match_dist = [getDist(ms) for ms in match_pos_set]

	if len(thresholds) == 1 and verbose :
		display.distHist(match_dist[0], thresholds[0])

	if return_matches : return match_pos_set, match_dist
	else : return match_dist


# Load all files
def getImagePairs(directory) :
	dir_path = "../../images/testsets/%s/" % directory
	pairs = [dir_path + p for p in os.listdir(dir_path) if len(p) <= 6]
	homographies = [getHomography(p) for p in pairs]
	return [([p + "_1.jpg", p + "_2.jpg"], h) for (p,h) in zip(pairs, homographies)]


def folderMatch(directory, dt, match_fun, thresholds, keypoint, descriptor) : 

	def flatten(l) :
		return list(itertools.chain.from_iterable(l))

	def nb_correct_matches(distances) :
		return len([d for d in distances if d <= dt])

	# Get image pairs of folder
	imagePairs = getImagePairs(directory)

	# Get a set of matches varied over image pairs (matrix of size N x T where N
	# is amount of image pairs and T is the amount of thresholds. Each element in
	# the matrix is a zipped list of matches and distances
	match_sets = numpy.array([numpy.array(match_fun(dt, p, h, thresholds, keypoint, descriptor)) for p, h in imagePairs]).T

	# Now collect an x axis with nb correct matches per threshold
	get_correct_matches = lambda match_row : [nb_correct_matches(d) for d in match_row]
	nb_correct_set = [get_correct_matches(row) for row in match_sets]

	# Now collect an x axis with nb correct matches per threshold
	get_total = lambda match_row : [len(d) for d in match_row]
	nb_total_set = [get_total(row) for row in match_sets]

	return nb_correct_set, nb_total_set


def getHomography(hom_path) :
	h = numpy.loadtxt(hom_path)
	return h


def getTestsetPaths(collection, index = None) :
	pairs = getImagePairs(collection)
	if index == None : return random.choice(pairs)
	else : return pairs[index]



def getPaths(orig_id = None, compare_id = 2, img_type = "graf", folder = "inria") :

	# Dictionary to load the files
	file_endings = {
		"GRAFFITI6" : "ppm",
		"bark" : "ppm",
		"bikes" : "ppm",
		"boat" : "pgm",
		"graf" : "ppm",
		"leuven" : "ppm",
		"trees" : "ppm",
		"ubc" : "ppm",
		"wall" : "ppm"
	}

	# Get File ending
	img_ending = file_endings.get(img_type,"pgm")

	# Get directory
	dir_path = "../../images/%s/%s/" % (folder, img_type)

	# Get id of first image in series
	i_first = int(fnmatch.filter(os.listdir(dir_path), "H[0-9]to[0-9]*")[0][1])
	
	# let i1 be original id if specified, else we'll set it to i_first
	i1 = i_first if orig_id == None else orig_id

	# Get image basename
	basename = fnmatch.filter(os.listdir(dir_path), "*[a-z]%s.%s" % (i1,img_ending))[0].split(str(i1))[0]
	i2 = compare_id
	p1 = "%s%s%i.%s" % (dir_path, basename, i1, img_ending)
	p2 = "%s%s%i.%s" % (dir_path, basename, i2, img_ending)
	paths = [p1, p2]

	def getH(i) :
		hom_name = filter(re.compile("H%ito%i(p|prec)?$" % (i_first, i)).match, os.listdir(dir_path))[0]
		hom_path = "%s%s" % (dir_path, hom_name)
		return getHomography(hom_path)

	# Load and parse homography
	if i_first == i1 : 
		hom = getH(i2)
	else :
		hom_1 = getH(i1)
		hom_2 = getH(i2)
		hom = hom_2.dot(numpy.linalg.inv(hom_1))

	return paths, hom



def getACRPaths(orig_id = None, compare_id = 2, img_type = "graf") :
	return getPaths(orig_id, compare_id, img_type, "acr")



def ratioMatch(paths, thresholds, options = {}) :

	keypoint_type		= options.get("keypoint_type", "SIFT")
	descriptor_type		= options.get("descriptor_type", "SIFT")

	# Get all feature points
	indices, ks, ds = features.getFeatures(paths, keypoint_type = keypoint_type, descriptor_type = descriptor_type)

	# Use cv2's matcher to get matching feature points
	ii, ss, uu = features.bfMatch(descriptor_type, ds[indices == 0], ds[indices == 1])
	#bfMatches = features.match("BRIEF", ds[ind == 0], ds[ind == 1])
	
	# Check that we have keypoints:
	if ss == None : return [[] for _ in thresholds]

	# Get matches in usual format
	def matchFromIndex(i,j) :
		return (features.getPosition(ks[indices == 0][i]), features.getPosition(ks[indices == 1][j]))

	# Now for each threshold test the uniqueness of the matches
	p = lambda t : [matchFromIndex(i,j) for i,j in enumerate(ii) if uu[i] < t] 
	match_set = [p(t) for t in thresholds]

	return numpy.array(match_set)


def standardMatch(paths, thresholds, options = {}) :

	keypoint_type		= options.get("keypoint_type", "SIFT")
	descriptor_type		= options.get("descriptor_type", "SIFT")

	# Get all feature points
	indices, ks, ds = features.getFeatures(paths, keypoint_type = keypoint_type, descriptor_type = descriptor_type)

	# Use cv2's matcher to get matching feature points
	jj, ss_j, uu_j = features.bfMatch(descriptor_type, ds[indices == 0], ds[indices == 1])
	ii, ss_i, uu_i = features.bfMatch(descriptor_type, ds[indices == 1], ds[indices == 0])
	#bfMatches = features.match("BRIEF", ds[ind == 0], ds[ind == 1])
	
	# Check that we have keypoints:
	if ss_j == None or ss_i == None : return [[] for _ in thresholds]

	# Get matches in usual format
	def matchFromIndex(i,j) :
		return (features.getPosition(ks[indices == 0][i]), features.getPosition(ks[indices == 1][j]))

	# See if matches go both ways
	bothways = [(i,j) for i,j in enumerate(jj) if ii[j] == i]

	# Now for each threshold test the uniqueness of the matches
	p = lambda t : [matchFromIndex(i,j) for i,j in bothways if uu_j[i] < t] 
	match_set = [p(t) for t in thresholds]

	return numpy.array(match_set)






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






def getDistinctMatches(matches, threshold = 5) :
	
	def distinctFrom(m, head) :
		a = numpy.linalg.norm(numpy.array(m[0]) - numpy.array(head[0]))
		b = numpy.linalg.norm(numpy.array(m[1]) - numpy.array(head[1]))
		return a > threshold or b > threshold
	
	if len(matches) < 1 : return []
	head = matches[0]
	tail = [m for m in matches[1:] if distinctFrom(m, head)]
	return [head] + getDistinctMatches(tail, threshold)



def clusterMatch(distance_threshold, paths, homography, thresholds, keypoint, descriptor) :
	return testMatch(
		paths, 
		homography, 
		clustermatch.match,
		verbose = False,
		options = {
			"prune_fun" : weightMatrix.pruneThreshold, 
			"prune_limit" : 2.5,
			"min_coherence" : 0.0,
			"thresholds" : thresholds,
			"split_limit" : 500,
			"cluster_prune_limit" : 1.5,
			"distance_threshold" : distance_threshold,
			"keypoint_type" : keypoint,
			"descriptor_type" : descriptor,
		})


def siftMatch(distance_threshold, paths, homography, thresholds, keypoint, descriptor) :
	return testMatch(
		paths,
		homography, 
		ratioMatch,
		verbose = False,
		options = {
			"thresholds" : thresholds,
			"distance_threshold" : distance_threshold,
			"keypoint_type" : keypoint,
			"descriptor_type" : descriptor,
		})


def mirrorMatch(distance_threshold, paths, homography, thresholds, keypoint, descriptor) :
	return testMatch(
		paths,
		homography, 
		mirrormatch.match,
		verbose = False,
		options = {
			"thresholds" : thresholds,
			"distance_threshold" : distance_threshold,
			"keypoint_type" : keypoint,
			"descriptor_type" : descriptor,
		})


def isoMatch(distance_threshold, paths, homography, thresholds, keypoint, descriptor) :
	return testMatch(
		paths,
		homography, 
		isomatch.match,
		verbose = False,
		options = {
			"thresholds" : thresholds,
			"distance_threshold" : distance_threshold,
			"keypoint_type" : keypoint,
			"descriptor_type" : descriptor,
		})
