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
import louvainmatch
import features
import weightMatrix
import display
import pylab
import fnmatch
import os
import re



####################################
#                                  #
#           Functions              #
#                                  #
####################################


def testMatch(paths, homography, match_fun, options = {}, verbose = True) :

	distinct_treshold = options.get("distinct_treshold", 5)
	distance_treshold = options.get("distance_treshold", 5)

	# Get matches
	matches = match_fun(paths, options)

	# Get distinct matches
	matches_distinct = getDistinctMatches(matches, distinct_treshold)

	# Get distances
	dist = [matchDistance(m1,m2,homography) for (m1,m2) in matches]
	dist_distinct = [matchDistance(m1,m2,homography) for (m1,m2) in matches_distinct]

	# Display result
	if verbose : display.distHist(dist, distance_treshold, dist_distinct)

	return matches, dist, dist_distinct


def getHomography(hom_path) :
	return numpy.array([map(lambda s : float(s.strip()), line.strip().split()) for i,line in enumerate(open(hom_path)) if i < 3])

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



def clusterMatch(paths, options = {}) :
	ms = louvainmatch.match(paths, options)
	prune = options.get("prune", False)
	if prune :
		return pruneMatches(ms)
	else :
		return ms



def isodataMatch(paths, options = {}) :
	ms =  isomatch.match(paths, options)
	prune = options.get("prune", False)
	if prune :
		return pruneMatches(ms)
	else :
		return ms



def standardMatch(paths, options = {}) :

	match_limit			= options.get("match_limit", 500)
	unique_treshold		= options.get("unique_treshold", 1.0)
	keypoint_type		= options.get("keypoint_type", "ORB")
	descriptor_type		= options.get("descriptor_type", "BRIEF")
	prune 				= options.get("prune", False)

	# Get all feature points
	indices, ks, ds = features.getFeatures(paths, keypoint_type = keypoint_type, descriptor_type = descriptor_type)

	# Use cv2's matcher to get matching feature points
	bfMatches = features.bfMatch(descriptor_type, ds[indices == 0], ds[indices == 1])
	#bfMatches = features.match("BRIEF", ds[ind == 0], ds[ind == 1])

	# Get matches in usual format
	def matchFromIndex(i,j) :
		return (features.getPosition(ks[indices == 0][i]), features.getPosition(ks[indices == 1][j]))

	match_score = [(matchFromIndex(j,i), s, u) for j,(i,s,u) in enumerate(zip(*bfMatches)) if i != None and u < unique_treshold]

	# Return empty list if there weren't any matches
	if len(match_score) == 0 : return []

	matches, scores, uniques = zip(*match_score)

	# Take only the n best
	top_n = numpy.argsort(scores)[0:match_limit]
	matches_top = numpy.array(matches)[top_n]

	if prune :
		return pruneMatches(matches_top)
	else :
		return matches_top



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



def getDistinctMatches(matches, treshold = 5) :
	
	def distinctFrom(m, head) :
		a = numpy.linalg.norm(numpy.array(m[0]) - numpy.array(head[0]))
		b = numpy.linalg.norm(numpy.array(m[1]) - numpy.array(head[1]))
		return a > treshold and b > treshold
	
	if len(matches) < 1 : return []
	head = matches[0]
	tail = [m for m in matches[1:] if distinctFrom(m, head)]
	return [head] + getDistinctMatches(tail, treshold)



