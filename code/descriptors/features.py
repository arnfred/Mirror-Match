"""
Python module for use with openCV to extract features and descriptors
and match them using variuos approaches, including SIFT and SURF

Some of this code is adapted from Jan Erik Solem's python wrapper 
(http://www.janeriksolem.net/2009/02/sift-python-implementation.html),
which in turn is adapted from the matlab code examples at
http://www.cs.ubc.ca/~lowe/keypoints/

Jonas Toft Arnfred, 2013-03-07
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import cv2
import numpy



####################################
#                                  #
#           Functions              #
#                                  #
####################################


def getFeature(feature_type, params = {}) :
	""" Returns a feature object from opencv with the parameters set
		input: feature_type [string] (the type of the feature)
		       params [dict] (Extra parameters for the method)
	"""
	# Get hold of method
	if not cv2.__dict__.has_key(feature_type) : raise NonExistantMethod(feature_type)
	feature = cv2.__dict__[feature_type]()

	# Set parameters
	for key, value in params : setattr(m, key, value)

	return feature



def getKeypoints(feature_type, image, params = {}) :
	""" Given the feature_type and an image, we return the keypoints for this image
		input: feature_type [string] (The feature we are using to extract keypoints)
		       image [numpy.ndarray] (The image format used by scipy and opencv)
		       params [dict] (Extra parameters for the method)
	"""
	# Get the feature
	feature = getFeature(feature_type, params)
	
	# Return feature points
	return feature.detect(image)



def getDescriptors(feature_type, image, keypoints) :
	""" Given a set of keypoints we convert them to descriptors using the method 
	    specified by the feature_type
		input: feature_type [string] (The feature we are using to extract keypoints)
		       image [numpy.ndarray] (The image format used by scipy and opencv)
			   keypoints [list of cv2.Keypoint] (The keypoints we want to encode
		       params [dict] (Extra parameters for the method)
	"""
	# Make sure the feature_type exists and if it does, load it
	if not cv2.__dict__.has_key(feature_type) : raise NonExistantMethod(feature_type)
	extractor = cv2.DescriptorExtractor_create(feature_type)

	# compute descriptors
	keypoints, descriptors = extractor.compute(image, keypoints)
	
	return descriptors



def match(D1, D2, ratio = 0.6) :
	""" for each descriptor in the first image, select its match to second image
		input: desc1 [numpy.ndarray] (matrix with descriptors for first image), 
			   desc2 [numpy.ndarray] (same for second image)
			   ratio [float] (The difference between the closest and second
							  closest keypoint match.)
		Adapted from http://www.janeriksolem.net/2009/02/sift-python-implementation.html
	"""
	# Return the first element if the ratio between first and second 
	# element is more than the ratio argument
	def bestMatch(row) :
		s = numpy.argsort(row)
		return s[0] if row[s[0]] < ratio*row[s[1]] else -1
	
	# Compute D1 * D2, and find approx dist with arccos
	T = D1.dot(D2.T)
	matchscores = [bestMatch(row) for row in numpy.arccos(T)]
	
	return matchscores



def loadImage(path) : 
	""" Given a path, an image will be loaded and converted to grayscale 
		input: path [string] (path to the image)
	"""
	# Try to read image, and if doesn't exist, throw exception
	img = cv2.imread(path)
	if (img == None) : raise NonExistantPath(path, "Image doesn't exist")

	# Convert to grayscale: First we convert the image to the L*u*v color space
	# and then return the luminance channel
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:,:,0]

	return img_gray


####################################
#                                  #
#           Exceptions             #
#                                  #
####################################

class NonExistantPath(Exception) :
	def __init__(self, path, msg = "") :
		self.path = path
		self.msg = msg

class NonExistantMethod(Exception) :
	def __init__(self, method, msg = "") :
		self.path = path
		self.msg = msg
