"""
Python module for matching two images using proposed mirror match 
algorithm. This is an improved version of unique matching

Jonas Toft Arnfred, 2013-05-23
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import features


####################################
#                                  #
#           Functions              #
#                                  #
####################################

def match(paths, options = {}) :

	# Returns the position of a match
	def matchPos(i,j) :
		return (features.getPosition(ks[i]), features.getPosition(ks[j]))

	# Get options
	keypoint_type		= options.get("keypoint_type", "SIFT")
	descriptor_type		= options.get("descriptor_type", "SIFT")
	include_keypoints	= options.get("include_keypoints", False)

	# Get all feature points
	indices, ks, ds = features.getFeatures(paths, keypoint_type = keypoint_type, descriptor_type = descriptor_type)

	# Use cv2's matcher to get matching feature points
	jj,ss,uu = features.bfMatch(descriptor_type, ds, ds, match_same=True)
	
	# See if matches go both ways
	bothways = [(i,j) for i,j in enumerate(jj) if indices[i] == 0 and indices[j] == 1 and jj[j] == i]

	# Define a function that given a threshold returns a set of matches
	def match_fun(threshold) :
		match_data = [(matchPos(i,j), uu[i], ss[i]) for i,j in bothways if uu[i] < threshold]

		if include_keypoints : 
			if len(match_data) == 0 : return [], [], [], (ks[indices == 0], ks[indices == 1])
			matches, ratios, scores = zip(*match_data)
			return matches, ratios, scores, (ks[indices == 0], ks[indices == 1])
		else : 
			if len(match_data) == 0 : return [], [], []
			matches, ratios, scores = zip(*match_data)
			return matches, ratios, scores

	return lambda t : match_fun(t)
