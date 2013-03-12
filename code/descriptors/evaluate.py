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
from itertools import combinations



def imageToImage(paths, keypoint_type, descriptor_type, score_fun = lambda i,s,u : numpy.mean(s)) :
	""" Compare every image with every other image, generating a few different scores
	    input: paths [List of Strings] Paths of all the images
		       keypoint_type [String] e.g "SURF" or "ORB" etc
		       keypoint_descriptor [String] e.g "SURF" or "ORB" etc
		       score_fun [(list(int), list(int/float), list(float)) -> float]
			     The score function take a list of indices, a list of scores (distance
				 between two descriptors), a list of scores (uniqueness of best/second
				 best score) and returns a floating point number.
	    output: [list of (boolean, score)] a list where the boolean is true if the images
		                                   where of the same person and false if not
	"""

	# Load all images
	images = map(f.loadImage, paths)

	# Get keypoints
	keypoints = map(lambda i : f.getKeypoints(keypoint_type, i), images)

	# Get descriptors
	descriptors = map(lambda i,k : f.getDescriptors(descriptor_type, i, k), images, keypoints)

	# Create a set of labels
	labels = map(f.getLabel, paths)

	# Return the scores labeled with a boolean to indicate if they are of same set
	return matchDescriptors(descriptors, labels, descriptor_type, score_fun)



def matchDescriptors(descriptors, labels, descriptor_type, score_fun) :

	def filterNone(l) : return [i for i in l if i != None]
	def getScore(D1, D2) :
		indices, scores, uniques = f.match(descriptor_type, D1, D2)
		noNones = map(filterNone, [indices, scores, uniques])
		return score_fun(*noNones)

	return [((l1 == l2), getScore(D1,D2))
			for ((D1, D2), (l1, l2)) 
			in zip( combinations(descriptors,2), combinations(labels,2) )]
