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
import collections
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
    verbose             = options.get("feature_verbose", False)
    feature_type        = options.get("feature_type", "L")
    filter_features     = options.get("filter_features", numpy.array([]))

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
               # Use custom made matcher
               options [dictionary] keypoint_type, max_kp
        out:   [list of cv2.Keypoint]
    """

    # Get amount of feature points
    keypoint_type		= options.get("keypoint_type", "SIFT")
    max_kp              = options.get("max_kp", 9999) 
    #kp_threshold        = options.get("kp_threshold", (0.04, 10.0))
    verbose             = options.get("feature_verbose", False)

    if verbose : print(":"),

    # Check if feature_type exists
    if not keypoint_type.upper() in supported_keypoint_types : raise NonExistantFeature(keypoint_type)

    # Get specific feature type
    if keypoint_type.upper() == "SIFT" :
        feature = cv2.SIFT()#nfeatures = max_kp, contrastThreshold = kp_threshold[0], edgeThreshold = kp_threshold[1])
    elif keypoint_type.upper() == "FAST" :
        feature = cv2.FastFeatureDetector()
    elif keypoint_type.upper() == "SURF" :
        feature = cv2.SURF()
    elif keypoint_type.upper() == "ORB" :
        feature = cv2.ORB()
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
    verbose             = options.get("feature_verbose", False)

    if verbose : print(";"),

    # Check if feature_type exists
    if not descriptor_type in supported_descriptor_types : raise NonExistantFeature(descriptor_type)

    feature = cv2.DescriptorExtractor_create(descriptor_type)

    # compute descriptors
    return feature.compute(image, keypoints)


# Map for the type of distance measure to use
def dist_map(descriptor_type) :
    dist_dict = {
        "SIFT"   : 'minkowski',
        "SURF"   : 'minkowski',
        "ORB"    : 'hamming',
        "BRISK"  : 'hamming',
        "BRIEF"  : 'hamming',
        "FREAK"  : 'hamming'
    }
    return dist_dict[descriptor_type]


def match(D, indices, options = {}) :

    # Default ratio
    default_ratio       = { "proposed" : "target", "baseline" : "both" }

    # Get options
    descriptor_type		= options.get("descriptor_type", "SIFT")
    verbose             = options.get("match_verbose", False) 
    tree_type           = options.get("tree_type", "ball")
    proposed            = options.get("ratio", default_ratio)["proposed"]
    baseline            = options.get("ratio", default_ratio)["baseline"]

    # Add options
    options['dist_metric'] = dist_map(descriptor_type)
    I = numpy.array(range(len(indices)))

    def split_D(split_fun) :
        for i in set(indices) :
            split = split_fun(i)
            yield (I[split], D[split])

    # split feature points up by image
    S_query = list(split_D(lambda i : indices == i))

    # Find matches within images
    if baseline == "query" :
        if verbose : print("\nCalculating query matches")
        T_query = [trees.init(D_im, tree_type, options) for (I_im, D_im) in S_query]
        M_query = [T(D_im, 2) for (I_im, D_im), T in zip(S_query, T_query)]

    # Find matches across images
    if proposed == "target" or baseline == "target" : 
        if verbose : print("\nCalculating target matches")
        S_target = list(split_D(lambda i : indices != i))
        T_target = [trees.init(D_im, tree_type, options) for (I_im, D_im) in S_target]
        M_target = [T(D_im, 2) for (I_im, D_im), T in zip(S_query, T_target)]

    # Find matches all across and within images
    if proposed in ["both","all"] or baseline in ["both","all"] :
        if verbose : print("\nCalculating all matches")
        S_all = [(I,D)]*len(S_query)
        T_all = trees.init(D, tree_type, options)
        M_all = [T_all(D_im, 3) for I_im, D_im in S_query]


    # Produce matches across two different trees
    def dual(M_proposed, M_baseline, S_proposed, overlap) :

        # The match indexes depending on overlap
        fst_proposed, fst_baseline = (int(overlap[0]), int(overlap[1]))

        # For each image ...
        data_all = zip(M_proposed, M_baseline, S_query, S_proposed)
        for matches_proposed, matches_baseline, (index_map_from, d_im), (index_map_to, d_im) in data_all :

            # For each match in this image ...
            data_im = zip(index_map_from, matches_proposed, matches_baseline)
            for index_from, (idx_proposed, dist_proposed), (idx_baseline, dist_baseline) in data_im :

                # Get the baseline score
                score_baseline = dist_baseline[fst_baseline]

                # For each match we make in the images
                for i in range(fst_proposed, len(idx_proposed)) :
                    index_to = index_map_to[idx_proposed[i]]
                    score_proposed = dist_proposed[i]
                    ratio = score_proposed / float( score_baseline )
                    yield (index_from, index_to), score_proposed, ratio


    # Produce matches within the same tree
    def single(M, S, overlap) :

        # The match indexes depending on overlap
        if overlap :    fst, snd = (1, 2)
        else :          fst, snd = (0, 1)

        for matches_im, (index_map_from, d_im), (index_map_to, d_im) in zip(M, S_query, S) :
            for index_from, (idx_array_to, dsc) in zip(index_map_from, matches_im) :
                index_to = index_map_to[idx_array_to[fst]]
                score = dsc[fst]
                ratio = score / float( dsc[snd] )
                yield (index_from, index_to), score, ratio

    #print(map(len, D_target))
    #print(map(len, I_target))
    #print(map(len, M_target))

    
    # Mirror match
    if proposed in ["both","all"] and baseline in ["both","all"] :
        return list(single(M_all, S_all, overlap = True))

    # Ratio match
    elif proposed == "target" and baseline == "target" :
        return list(single(M_target, S_target, overlap = False))

    # query Strict == query Both for most parts
    elif proposed == "target" and baseline == "query" :
        return list(dual(M_target, M_query, S_target, overlap = (False, True)))

    # query Both == query Strict for most parts
    elif proposed in ["both","all"] and baseline == "query" :
        return list(dual(M_all, M_query, S_all, overlap = (True, True))) # query always overlaps with itquery

    # Ratio Ext
    elif proposed in ["both","all"] and baseline == "target" :
        return list(dual(M_all, M_target, S_all, overlap = (True, True)))

    # Weird Match == Mirror Match for most parts
    elif proposed == "target" and baseline in ["both","all"] :
        # Why 2? Well, #0 is the same point, and #1 is either the match we found or the best match
        # in the target image
        return list(dual(M_target, M_all, S_target, overlap = (False, 2)))

    else :
        raise Exception("Unknown ratio combination: proposed = '%s', baseline = '%s'" % (proposed, baseline))



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



def loadImage(path, feature_type = 'L') : 
    """ Given a path, an image will be loaded and converted to grayscale
        input: path [string] (path to the image)
               feature_type [string] (either 'L', 'u' or 'v' for luminance or colors)
        out:   [numpy.ndarray] cv2 representation of image in one channel
    """

    # We can return the loaded color image
    if feature_type == "all" :
        return cv2.imread(path)

    # If not then we return specific channel
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


def flatten(it):
    for x in it:
        if (isinstance(x, collections.Iterable) and not isinstance(x, str)):
            for y in x:
                yield y
        else:
            yield x

####################################
#                                  #
#           Exceptions             #
#                                  #
####################################

class NonExistantPath(Exception) :
    def __init__(self, path, msg = "") :
        print(path)
        self.path = path
        self.msg = msg
    def __str__(self):
        return repr("The path %s does not exist: %s" % (self.path, self.msg))

class NonExistantFeature(Exception) :
    def __init__(self, method, msg = "") :
        self.method = method
        self.msg = msg
    def __str__(self):
        return repr("The feature %s does not exist: %s" % (self.method, self.msg))


