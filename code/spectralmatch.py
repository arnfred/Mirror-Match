"""
Python module for matching two images using spectral matching.

Based on:
Leordeanu, Marius, and Martial Hebert. "A spectral technique for correspondence problems using pairwise constraints." Computer Vision, 2005. ICCV 2005. Tenth IEEE International Conference on. Vol. 2. IEEE, 2005.
http://repository.cmu.edu/cgi/viewcontent.cgi?article=1362&context=robotics

Jonas Toft Arnfred, 2013-07-03
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import features
import display
import numpy
import itertools
import matching


####################################
#                                  #
#           Functions              #
#                                  #
####################################


def match(paths, thresholds, options ={}) :

	# Get parameters
	keypoint_type = options.get("keypoint_type", "SIFT")
	descriptor_type = options.get("descriptor_type", "SIFT")
	verbose = options.get("verbose", False)
	affinity_fun = options.get("affinity", affinity_simple)
	sigma = options.get("affinity", 50)
	nb_sources = options.get("nb_sources", 500)
	nb_matches = options.get("nb_matches", 100)

	# Get all feature points
	indices, ks, ds = features.getFeatures(paths, keypoint_type = keypoint_type, descriptor_type = descriptor_type)
	(pos_im1, pos_im2) = (features.getPositions(ks[indices == 0]), features.getPositions(ks[indices == 1]))

	match_data = features.bfMatch(descriptor_type, ds[indices == 0], ds[indices == 1])

	# Get threshold for n best matches
	threshold = numpy.sort(match_data[1])[nb_sources]

	# Get matches and scores
	matches = numpy.array([(pos_im1[i], pos_im2[j]) for (i,(j,s,r)) in enumerate(zip(*match_data)) if s < threshold])
	scores = numpy.array([s for (j,s,r) in zip(*match_data) if r < threshold])

	# Get affinity matrix
	M = affinity_matrix(matches, scores, affinity_fun, sigma)

	# Get eigenvectors
	eigvals, eigvecs = numpy.linalg.eigh(M)
	x_star = eigvecs[:,eigvals.argsort()[-1]]

	# For some reason I get an all-negative eigenvector at times. I'm just making sure it's positive here
	if x_star[0] < 0 : x_star = x_star * -1
		
	# Pick best n
	best_m = numpy.array(matches)[numpy.argsort(x_star)[(-1*nb_matches):]]
	if verbose : 
		images = map(features.loadImage, paths)
		display.matchPoints(images[0], images[1], best_m)
		print("Best %i matches picked using geometric constraints" % nb_matches)

	return best_m


# Fill in the affinity matrix
def affinity_matrix(matches, scores, affinity_fun, sigma) :
	# Construct matrix
	M = numpy.zeros((len(matches), len(matches)))
	for i,m1 in enumerate(matches) :
		for j,m2 in enumerate(matches) :
			if i == j :
				M[i,j] = 4.5*(1 - (scores[i] / numpy.max(scores)))
			else :
				M[i,j] = affinity_fun(m1,m2, sigma)
	return M


# Affinity function
def affinity_simple(m1,m2, sigma) :
	d0 = numpy.linalg.norm(m1[0] - m2[0])
	d1 = numpy.linalg.norm(m1[1] - m2[1])
	M_ab = 4.5 - ((d0 - d1) ** 2) / (2*(sigma**2))
	if numpy.abs(d0 - d1) < 3*sigma :
		return M_ab
	else :
		return 0.0
