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


def match(paths, options ={}) :

	# Get parameters
	affinity_fun = options.get("affinity", affinity_simple)
	matching_fun = options.get("match_fun", getMatches)
	sigma = options.get("affinity", 50)
	keep_ratio = options.get("keep_ratio", 0.5)
	verbose = options.get("verbose", False)

	matches, scores, ratios = matching_fun(paths, options)
	nb_matches = len(matches) * keep_ratio
	if (nb_matches < 2) : return [[] for t in thresholds]

	# Get affinity matrix
	M = affinity_matrix(matches, scores, affinity_fun, sigma)

	# Get eigenvectors
	eigvals, eigvecs = numpy.linalg.eigh(M)
	x_star = eigvecs[:,eigvals.argsort()[-1]]

	# For some reason I get an all-negative eigenvector at times. I'm just making sure it's positive here
	if x_star[0] < 0 : x_star = x_star * -1
		
	# Pick best half matches
	best_m = numpy.array(matches)[numpy.argsort(x_star)[int(-1*nb_matches):]]
	best_ratios = numpy.array(ratios)[numpy.argsort(x_star)[int(-1*nb_matches):]]
	best_scores = numpy.array(scores)[numpy.argsort(x_star)[int(-1*nb_matches):]]

	if verbose : 
		images = map(features.loadImage, paths)
		display.matchPoints(images[0], images[1], best_m)
		print("Best %i matches picked using geometric constraints" % nb_matches)
		print("The score of x_star is: %.2f" % (x_star.T.dot(M).dot(x_star)))


	def match_fun(threshold) :
		match_data = [(pos, u, s) for (pos,u,s) in zip(best_m, best_ratios, best_scores) if u < threshold]
		if len(match_data) == 0 : return [], [], []
		matches, ratios, scores = zip(*match_data)

		return matches, ratios, scores

	return lambda t : match_fun(t)


def matchAlt(paths, options = {}) :
	# Get parameters
	affinity_fun = options.get("affinity", affinity_simple)
	matching_fun = options.get("match_fun", getMatches)
	sigma = options.get("affinity", 50)
	verbose = options.get("verbose", False)
	threshold = options.get("threshold", 0.94)

	matches, scores, ratios = matching_fun(paths, options)
	# If we don't have any matches, then return a function that always returns an empty list

	if len(matches) > 0 :

		# Get affinity matrix
		M = affinity_matrix(matches, scores, affinity_fun, sigma)

		# Get eigenvectors
		eigvals, eigvecs = numpy.linalg.eigh(M)
		x_star = eigvecs[:,eigvals.argsort()[-1]]

		# For some reason I get an all-negative eigenvector at times. I'm just making sure it's positive here
		if x_star[0] < 0 : x_star = x_star * -1
		
	def match_fun(threshold) :

		# Pick all matches higher than the threshold ratio
		nb_matches = len(matches) * threshold
		if nb_matches == 0 : return [], [], []

		best_m = numpy.array(matches)[numpy.argsort(x_star)[int(-1*nb_matches):]]
		best_ratios = numpy.array(ratios)[numpy.argsort(x_star)[int(-1*nb_matches):]]
		best_scores = numpy.array(scores)[numpy.argsort(x_star)[int(-1*nb_matches):]]

		return best_m, best_ratios, best_scores

	return lambda t : match_fun(t)



def getMatches(paths, options = {}) :
	match_fun = match(paths, options)
	return match_fun(1.0)


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
