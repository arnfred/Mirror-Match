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
import preprocessing



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


def getFeatures(paths, keypoint_type = "ORB", descriptor_type = "BRIEF", size=32) :
	""" Given a list of paths to images, the function returns a list of 
	    descriptors and keypoints
		Input: paths [list of strings] The paths to the images we are using
		Out:   [pair of list of descriptors and list of keypoints]
	"""
	# Get all images and labels
	labels = map(getLabel, paths)
	images = map(loadImage, paths)

	# Get feature descriptors
	keypoints = [getKeypoints(keypoint_type, im) for im in images]
	#for k in numpy.concatenate(keypoints) :
	#	k.size = float(size)
	#keypoints = [getORBKeypoints(im, size) for im in images]
	data = [getDescriptors(descriptor_type, im, k) for (im, k) in zip(images, keypoints)]
	keypoints, descriptors = zip(*data)

	# Check that we could get descriptors for all images
	if sum(map(lambda d : d == None, descriptors)) > 0 : return (None, None, None)
	indices = [l for i,n in zip(range(len(labels)), map(len, descriptors)) for l in [i]*n]
	return (indices, numpy.concatenate(keypoints), numpy.concatenate(descriptors))



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


def getORBKeypoints(image, size=[32]) :
	if isinstance(size, (int, long)) : size = [size]
	#o = cv2.ORB(nfeatures=2000, scaleFactor=1.06, nlevels=15, edgeThreshold=n, patchSize=n)
	os = [cv2.ORB(nfeatures=2000, scaleFactor=1.06, nlevels=15, edgeThreshold=s, patchSize=s) for s in size]
	kpts = [kpt for o in os for kpt in o.detect(image)]
	#o = cv2.ORB(nfeatures=2000, scaleFactor=1.06, nlevels=12, edgeThreshold=35, patchSize=35)
	return kpts


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
	
	return keypoints, descriptors



def match(descriptor_type, D1, D2) :
	""" for each descriptor in the first image, select its match in the second image
		input: descriptor_type [string] (The feature we are using to extract keypoints)
			   D1 [numpy.ndarray] (matrix of length n x k with descriptors for first image) 
			   D2 [numpy.ndarray] (matrix of length m x k with descriptors for second image)
			   ratio [float] (The difference between the closest and second
							  closest keypoint match.)
		out:   [list of floats] (list of length n with index of corresponding keypoint in second 
								 image if any and None if not)
	"""
	# def bestMatch(row) :
	# 	s = numpy.argsort(row)
	# 	return (s[0], row[s[0]]) if row[s[0]] < ratio*row[s[1]] else None


	dist_fun_map = {
		"SIFT"   : angleDist,
		"SURF"   : angleDist,
		"ORB"    : hammingDist,
		"BRISK"  : hammingDist,
		"BRIEF"  : hammingDist,
		"FREAK"  : hammingDist
	}

	# Get distFun
	dist_fun = dist_fun_map[descriptor_type]

	def getData(row) :
		ranking = numpy.argsort(row)
		# The index of the best and second best match
		i,j = ranking[0], ranking[1]
		# The score of the best and second best match
		s,t = row[i], row[j]
		# Uniqueness: The ration between the best and second best match
		u = s / (1.0 * t)
		return (i,s,u)

	T = dist_fun(D1,D2)
	m1 = [getData(row) for row in T]
	m2 = [getData(row) for row in T.T]

	m2_indices = zip(*m2)[0]

	data = [(i,s,u) if index == m2_indices[i] else (None,None,None) 
			  for ((i,s,u), index) in zip(m1, range(len(m1)))]

	return zip(*data)
	#T = dist_fun(D1,D2)
	#matchscores = [bestMatch(row) for row in T]
	#return matchscores



def bfMatch(descriptor_type, D1, D2) :

	# Map for the type of distance measure to use
	dist_map = {
		"SIFT"   : cv2.NORM_L2,
		"SURF"   : cv2.NORM_L2,
		"ORB"    : cv2.NORM_HAMMING,
		"BRISK"  : cv2.NORM_HAMMING,
		"BRIEF"  : cv2.NORM_HAMMING,
		"FREAK"  : cv2.NORM_HAMMING
	}

	# Map for the type of the data in the array
	type_map = {
		"SIFT"   : numpy.float32,
		"SURF"   : numpy.float32,
		"ORB"    : numpy.uint8,
		"BRISK"  : numpy.uint8,
		"BRIEF"  : numpy.uint8,
		"FREAK"  : numpy.uint8
	}

	dtype = type_map[descriptor_type]
	dist = dist_map[descriptor_type]

	# Now get BFMatcher
	matcher = cv2.BFMatcher(dist)

	# Make sure the array is encoded properly
	query = numpy.array(D1, dtype = dtype)
	train = numpy.array(D2, dtype = dtype)

	# Find nearest neighbor
	matches_qt = matcher.knnMatch(query, train, k=2)
	#matches_tq = bf.knnMatch(train, query, k=2)

	# Convert result
	data = [(ms[0].trainIdx, ms[0].distance, ms[0].distance*1.0/ms[1].distance) 
				for ms in matches_qt if len(ms) == 2]

	if len(data) == 0 : return (None, None, None)

	return zip(*data)



# Compute D1 * D2.T, and find approx dist with arccos
def angleDist(D1, D2) : return numpy.arccos(D1.dot(D2.T))



# Compute hamming distance
# CHECK weightMatrix.py for much faster method
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

	# Preprocess image
	#img_processed = preprocessing.faceNorm(img_gray)

	return img_gray



def getLabel(path) : return " ".join(path.split("/")[-1].split("_")[0:-1])


def getPosition(keypoint) :
	return (keypoint.pt[0], keypoint.pt[1])

def getPositions(keypoints) : 
	return map(getPosition, keypoints)


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
