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
import trees
from sklearn.neighbors.ball_tree import BallTree



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


def getFeatures(paths, options = {}) :
    """ Given a list of paths to images, the function returns a list of 
        descriptors and keypoints
        Input: paths [list of strings] The paths to the images we are using
               options [dictionary of options] shuffle, group, max_kp, keypoint_type, descriptor_type
        Out:   [pair of list of descriptors and list of keypoints]
    """

    # Get options
    shuffle             = options.get("shuffle", False)
    verbose             = options.get("verbose", False)
    feature_type        = options.get("feature_type", "L")

    # Get all images
    images = map(lambda i : loadImage(i, feature_type), paths)

    # Find feature points
    if shuffle :
        keypoints = [numpy.array(getKeypoints(im, options)) for im in images]
        map(numpy.random.shuffle, keypoints)
    else :
        keypoints = [numpy.array(getKeypoints(im, options)) for im in images]

    if verbose : print("\nKeypoints collected: %s" % str(map(len,keypoints)))

    # Describe featre points
    data = [getDescriptors(im, k, options) for (im, k) in zip(images, keypoints)]
    keypoints, descriptors = zip(*data)

    if verbose : print("\nDescriptors calculated: %s" % str(map(len,descriptors)))

    # Check that we could get descriptors for all images (TODO: throw exception?)
    if sum(map(lambda d : d == None, descriptors)) > 0 : return (None, None, None)

    # Create a list of indices
    indices = [numpy.array([i]*n) for i,n in zip(range(len(paths)), map(len, descriptors))]

    return numpy.concatenate(indices), numpy.concatenate(keypoints), numpy.concatenate(descriptors)



def getKeypoints(image, options = {}) :
    """ Given the feature_type and an image, we return the keypoints for this image
        input: descriptor_type [string] (The feature we are using to extract keypoints)
               image [numpy.ndarray] (The image format used by scipy and opencv)
               params [dict] (Extra parameters for the method)
               options [dictionary] keypoint_type, max_kp
        out:   [list of cv2.Keypoint]
    """

    # Get amount of feature points
    keypoint_type		= options.get("keypoint_type", "SIFT")
    max_kp              = options.get("max_kp", 9999) 
    kp_threshold        = options.get("kp_threshold", (0.05, 0.3))
    verbose             = options.get("verbose", False)

    if verbose : print(":"),

    # Check if feature_type exists
    if not keypoint_type.upper() in supported_keypoint_types : raise NonExistantFeature(keypoint_type)

    # Get specific feature type
    if keypoint_type.upper() == "SIFT" :
        feature = cv2.SIFT(nfeatures = max_kp, contrastThreshold = kp_threshold[0], edgeThreshold = kp_threshold[1])
    elif keypoint_type.upper() == "FAST" :
        feature = cv2.FastFeatureDetector(kp_threshold[0])
    elif keypoint_type.upper() == "SURF" :
        feature = cv2.ORB(nfeatures = max_kp, hesssianThreshold = kp_threshold[0])
    elif keypoint_type.upper() == "ORB" :
        feature = cv2.SURF(nfeatures = max_kp)
    else :
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


def getDescriptors(image, keypoints, options) :
    """ Given a set of keypoints we convert them to descriptors using the method 
        specified by the feature_type
        input: descriptor_type [string] (The feature we are using to extract keypoints)
               image [numpy.ndarray] (The image format used by scipy and opencv)
               keypoints [list of cv2.Keypoint] (The keypoints we want to encode)
               params [dict] (Extra parameters for the method)
        out:   [numpy.ndarray] (matrix of size n x 64 where n is the number of keypoints)
    """

    # Get options
    descriptor_type		= options.get("descriptor_type", "SIFT")
    verbose             = options.get("verbose", False)

    if verbose : print(";"),

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
        u = s / float(t)
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



def ballMatch(D1, D2, options = {}) :

    # Get options
    match_same          = options.get("match_same", False) 
    descriptor_type		= options.get("descriptor_type", "SIFT")
    leaf_size   		= options.get("leaf_size", 10)

    # Map for the type of distance measure to use
    dist_map = {
        "SIFT"   : 'minkowski',
        "SURF"   : 'minkowski',
        "ORB"    : 'hamming',
        "BRISK"  : 'hamming',
        "BRIEF"  : 'hamming',
        "FREAK"  : 'hamming'
    }

    # Construct ball tree
    tree = BallTree(D2, leaf_size=leaf_size, metric=dist_map[descriptor_type])

    # For each descriptor in D2, query tree
    def query() :
        for i, descriptor in enumerate(D1) :
            if match_same :
                (dist, index) = tree.query(descriptor, k=3)
                yield (i, index[0][1]), dist[0][1], dist[0][1]/float(dist[0][2])
            else :
                (dist, index) = tree.query(descriptor, k=2)
                yield (i, index[0][0]), dist[0][0], dist[0][0]/float(dist[0][1])

    # Return matches
    return list(query())



def mirrorMatch(D_all, indices, options = {}) :

    # Get options
    leaf_size           = options.get("leaf_size", 10) 
    descriptor_type		= options.get("descriptor_type", "SIFT")
    verbose             = options.get("verbose", False) 
    neighbors           = options.get("neighbors", 2)
    tree_type           = options.get("tree_type", "ball")

    # Map for the type of distance measure to use
    dist_map = {
        "SIFT"   : 'minkowski',
        "SURF"   : 'minkowski',
        "ORB"    : 'hamming',
        "BRISK"  : 'hamming',
        "BRIEF"  : 'hamming',
        "FREAK"  : 'hamming'
    }

    tree_options = {
            "leaf_size" : leaf_size,
            "dist_metric" : dist_map[descriptor_type],
            "verbose" : True,
    }

    # Construct D_list
    D_list = [D_all[indices == i] for i in range(len(set(indices)))]

    # Construct main ball tree
    tree = trees.init(D_all, tree_type, tree_options)
    #tree = BallTree(D_all, leaf_size=leaf_size, metric=dist_map[descriptor_type])

    # Construct minor trees
    forest = [trees.init(D, tree_type, tree_options) for D in D_list]
    #forest = [BallTree(D, leaf_size=leaf_size, metric=dist_map[descriptor_type]) for D in D_list]

    # Query trees for distances
    def query(D, T) :
        if verbose : print("."),
        return [T(d, k = neighbors) for d in D]

    Q_all = [tree(D, neighbors) for D in D_list]
    if verbose : print("\n")
    Q_self = [T(D, neighbors) for D, T in zip(D_list, forest)]

    # Now produce matches
    def matches() :
        for im_all, im_self in zip(Q_all, Q_self) :
            for (i_all, d_all), (i_self, d_self) in zip(im_all, im_self) :
                best_self = d_self[1]
                index_self = i_all[0]
                for i in range(1, len(i_all)) :
                    index_other = i_all[i]
                    score = d_all[i]
                    ratio = score / float( best_self )
                    yield (index_self, index_other), score, ratio

    # Return matches
    return list(matches())




def bfMatch(D1, D2, options = {}) :

    # Get options
    match_same          = options.get("match_same", False) 
    descriptor_type		= options.get("descriptor_type", "SIFT")

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
    matches_qt = matcher.knnMatch(query, train, k=3)
    #matches_tq = bf.knnMatch(train, query, k=2)

    first = 1 if match_same else 0
    second = 2 if match_same else 1

    # Convert result
    data = [(ms[first].trainIdx, ms[first].distance, float(ms[first].distance)/(ms[second].distance + 0.00001)) 
                for ms in matches_qt if len(ms) == 3]

    if len(data) == 0 : return (None, None, None)

    return [((i,j),s,u) for i, (j,s,u) in enumerate(data)]



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




def loadImage(path, feature_type = 'L') : 
    """ Given a path, an image will be loaded and converted to grayscale
        input: path [string] (path to the image)
               feature_type [string] (either 'L', 'u' or 'v' for luminance or colors)
        out:   [numpy.ndarray] cv2 representation of image in one channel
    """

    feature_index = { 'L' : 0, 'u' : 1, 'v' : 2}[feature_type]

    # Try to read image, and if doesn't exist, throw exception
    img = cv2.imread(path)
    if (img == None) : raise NonExistantPath(path, "Image doesn't exist")

    # Convert to grayscale: First we convert the image to the L*u*v color space
    # and then return the luminance channel
    img_LUV = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    img_gray = img_LUV[:,:,feature_index]

    def norm(im) :
        return numpy.array(255 * ((im - numpy.min(im)) / float(numpy.max(im) - numpy.min(im))) + 0.5, dtype=numpy.uint8)

    return norm(img_gray)


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
        import trees
        self.msg = msg
