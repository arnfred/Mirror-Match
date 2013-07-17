"""
Python module for matching two images using MM. This module is made for 
Near Duplicate Detection experiment.

Jonas Toft Arnfred, 2013-05-22
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import numpy
import features


####################################
#                                  #
#           Functions              #
#                                  #
####################################


def calcSIFTmatching(d_array_a, d_array_b, matching_th = None, options = {}) : 
	
	# Get parameters
	keypoint_type = options.get("keypoint_type", "SIFT")
	descriptor_type = options.get("descriptor_type", "SIFT")

	threshold = 0.64

	# Get all feature points
	#indices, ks, ds = features.getFeatures(paths, keypoint_type = keypoint_type, descriptor_type = descriptor_type)
	ds = numpy.concatenate((d_array_a[:,:128], d_array_b[:,:128]))
	indices = numpy.concatenate((numpy.zeros(d_array_a.shape[0]), numpy.ones(d_array_b.shape[0])))

	# Use cv2's matcher to get matching feature points
	jj,ss,uu = features.bfMatch(descriptor_type, ds, ds, match_same=True)
	
	# See if matches go both ways
	bothways = [(i,j) for i,j in enumerate(jj) if indices[i] == 0 and indices[j] == 1 and jj[j] == i]

	# Now for each threshold test the uniqueness of the matches
	# Returns the position of a match
	def matchPos(i,j) :
		return ()

	matches = [(i,j) for i,j in bothways if uu[i] < threshold]

	return len(matches) / float(indices.shape[0])

