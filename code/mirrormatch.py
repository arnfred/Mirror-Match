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

def match(paths, thresholds, options = {}) :

	# Returns the position of a match
	def matchPos(i,j) :
		return (features.getPosition(ks[i]), features.getPosition(ks[j]))

	# Get options
	keypoint_type		= options.get("keypoint_type", "ORB")
	descriptor_type		= options.get("descriptor_type", "BRIEF")

	# Get all feature points
	indices, ks, ds = features.getFeatures(paths, keypoint_type = keypoint_type, descriptor_type = descriptor_type)

	# Use cv2's matcher to get matching feature points
	jj,ss,uu = features.bfMatch(descriptor_type, ds, ds, match_same=True)
	
	# See if matches go both ways
	bothways = [(i,j) for i,j in enumerate(jj) if indices[i] == 0 and indices[j] == 1 and jj[j] == i]

	# Now for each threshold test the uniqueness of the matches
	p = lambda t : [matchPos(i,j) for i,j in bothways if uu[i] < t] 
	match_set = [p(t) for t in thresholds]

	return match_set
