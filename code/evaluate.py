"""
Python module for use with openCV to do experiments and display the results

Jonas Toft Arnfred, 2013-03-08
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import cv2
import pylab
import numpy
import display
import features as f
import sys
import math
from itertools import combinations



####################################
#                                  #
#           Functions              #
#                                  #
####################################

def imageToImage(images, paths, keypoint_type, descriptor_type, score_fun = lambda i,s,u : numpy.mean(s)) :
	""" Compare every image with every other image, generating a few different scores
	    input: images [List of nparrays] all the images
	           labels [List of Strings] labels of all the images
		       keypoint_type [String] e.g "SURF" or "ORB" etc
		       keypoint_descriptor [String] e.g "SURF" or "ORB" etc
		       score_fun [(list(int), list(int/float), list(float)) -> float]
			     The score function take a list of indices, a list of scores (distance
				 between two descriptors), a list of scores (uniqueness of best/second
				 best score) and returns a floating point number.
	    output: [list of (boolean, score)] a list where the boolean is true if the images
		                                   where of the same person and false if not
	"""

	# Get keypoints
	#keypoints = map(lambda i : f.getKeypoints(keypoint_type, i), images)

	# Get descriptors
	#data = map(lambda i,k : f.getDescriptors(descriptor_type, i, k), images, keypoints)
	data = [f.getFeatures([p],keypoint_type, descriptor_type) for p in paths]

	indices, keypoints, descriptors = zip(*data)

	# Return the scores labeled with a boolean to indicate if they are of same set
	return matchDescriptors(descriptors, paths, descriptor_type, score_fun)


def matchDescriptors(descriptors, paths, descriptor_type, score_fun) :

	default_map = {
		"SIFT"   : 1,
		"SURF"   : 1,
		"ORB"    : 100,
		"BRISK"  : 100,
		"BRIEF"  : 100,
		"FREAK"  : 100
	}

	def filterNone(l) : return [i for i in l if i != None]
	def getScore(D1, D2, i) :
		if D1 == None or D2 == None : return default_map[descriptor_type]
		indices, scores, uniques = f.bfMatch(descriptor_type, D1, D2)
		if indices == None : return default_map[descriptor_type]

		noNones = map(filterNone, [indices, scores, uniques])
		return score_fun(*noNones)

	# Get all pairings of descriptors and labels
	desc_pairs = list(combinations(descriptors,2))
	path_pairs = list(combinations(paths,2))

	# Print status
	print("=") * (int(len(path_pairs) / 100) +1)

	score = [((f.getLabel(p1) == f.getLabel(p2)), getScore(D1, D2, i), (p1, p2))
			for ((D1, D2), (p1, p2), i)
			in zip( desc_pairs, path_pairs, range(len(path_pairs)) )]

	# Add a newline
	print("")

	return score
