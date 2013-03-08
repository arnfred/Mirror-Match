"""
Python module for use with openCV to extract features and descriptors
and match them using variuos approaches, including SIFT and SURF

Some of this code is inspired by Jan Erik Solem's python wrapper 
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
#           Attributes             #
#                                  #
####################################

supported_keypoint_types = ["FAST","STAR","SIFT","SURF","ORB","MSER","BRISK","GFTT","HARRIS","Dense","SimpleBlob"]
supported_descriptor_types = ["SIFT","SURF","ORB","BRISK","BRIEF","FREAK"]



####################################
#                                  #
#           Functions              #
#                                  #
####################################

def getKeypoints(keypoint_type, image, params = {}) :
	""" Given the feature_type and an image, we return the keypoints for this image
		input: descriptor_type [string] (The feature we are using to extract keypoints)
		       image [numpy.ndarray] (The image format used by scipy and opencv)
		       params [dict] (Extra parameters for the method)
		out:   [list of cv2.Keypoint]
	"""
	# Check if feature_type exists
	if not keypoint_type in supported_keypoint_types : raise NonExistantFeature(keypoint_type)

	# Get the feature
	feature = cv2.FeatureDetector_create(keypoint_type)
	
	# Return feature points
	return feature.detect(image)



def getDescriptors(descriptor_type, image, keypoints) :
	""" Given a set of keypoints we convert them to descriptors using the method 
	    specified by the feature_type
		input: descriptor_type [string] (The feature we are using to extract keypoints)
		       image [numpy.ndarray] (The image format used by scipy and opencv)
			   keypoints [list of cv2.Keypoint] (The keypoints we want to encode)
		       params [dict] (Extra parameters for the method)
		out:   [numpy.ndarray] (matrix of size n x 64 where n is the number of keypoints)
	"""
	# Check if feature_type exists
	if not descriptor_type in supported_descriptor_types : raise NonExistantFeature(descriptor_type)

	feature = cv2.DescriptorExtractor_create(descriptor_type)

	# compute descriptors
	keypoints, descriptors = feature.compute(image, keypoints)
	
	return descriptors



def match(distFun, D1, D2, ratio = 0.6) :
	""" for each descriptor in the first image, select its match in the second image
		input: distFun [numpy.ndarray] (given D1 of size n x k and D2 of size m x k this 
		                                function must return matrix of size m x n with all
										distances between rows of D1 and D2)
			   D1 [numpy.ndarray] (matrix of length n x k with descriptors for first image) 
			   D2 [numpy.ndarray] (matrix of length m x k with descriptors for second image)
			   ratio [float] (The difference between the closest and second
							  closest keypoint match.)
		out:   [list of floats] (list of length n with index of corresponding keypoint in second 
								 image if any and None if not)
	"""
	def bestMatch(row) :
		s = numpy.argsort(row)
		return s[0] if row[s[0]] < ratio*row[s[1]] else None

	T = distFun(D1,D2)
	m1 = [numpy.argmin(row) for row in T]
	m2 = [numpy.argmin(row) for row in T.T]


	return [pos if index == m2[pos] else None for (pos, index) in zip(m1, range(len(m1)))]
	#T = distFun(D1,D2)
	#matchscores = [bestMatch(row) for row in T]
	#return matchscores



# Compute D1 * D2.T, and find approx dist with arccos
def angleDist(D1, D2) : return numpy.arccos(D1.dot(D2.T))



# Compute hamming distance
def hammingDist(D1, D2) :
	# Fast function for computing hamming distance
	# n and m should both be integers
	def hammingDistInt(n,m):
		k = n ^ m
		count = 0
		while(k):
			k &= k - 1
			count += 1
		return(count)

	# Vectorizing the hammingDistInt function to take arrays
	hm = numpy.vectorize(hammingDistInt)

	# Record size and initialize data
	n = D1.shape[0]
	m = D2.shape[0]
	result = numpy.zeros((n,m), 'uint8')

	# Fill each spot in the resulting matrix with the distance
	for i in range(n) : 
		for j in range(m) : 
			result[i,j] = sum(hm(D1[i], D2[j]))
	
	return result




def loadImage(path) : 
	""" Given a path, an image will be loaded and converted to grayscale
		input: path [string] (path to the image)
		out:   [numpy.ndarray] cv2 representation of image in one channel
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

class NonExistantFeature(Exception) :
	def __init__(self, method, msg = "") :
		self.method = method
		self.msg = msg
