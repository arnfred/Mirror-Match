"""
Python module for matching two images using various techniques including:
- Louvain clustering of feature points by similarity
- Isodata clustering of feature points by geometry
- Nearest neighbor matching of feature points by similarity
- As above but filtering out points that aren't unique

Jonas Toft Arnfred, 2013-04-22
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import numpy
import fnmatch
import os
import re
import itertools
import random
import isomatch
import clustermatch
import isomatch
import mirrormatch
import ballmatch
import spectralmatch
import ratiomatch
import features
import weightMatrix
import display
import trees



####################################
#                                  #
#           Functions              #
#                                  #
####################################


def evaluate(match_fun, paths, thresholds, homography, options) :

    # Get distance_threshold
    distance_threshold      = options.get("distance_threshold", 5)
    verbose                 = options.get("evaluate_verbose", False)
    metric                  = options.get("metric", "match_count")

    # Get matches
    positions, distances, ratios, image_indices, indices = match_fun(paths, options = options)(9999)

    if verbose :
        path = lambda p : "/".join(p.split("/")[-2:])
        print("Found %i matches for images (%s, %s)" % (len(ratios), path(paths[0]), path(paths[1])))

    # Get distances
    distances = [matchDistance(p1, p2, homography) for (p1, p2) in positions]

    if verbose :
        print("Found %i distances" % (len(distances)))

    # Collect precision per ratio_threshold
    if metric == "match_count" :
        def get_count(t, dt) : return len([1 for d,r in zip(distances, ratios) if r <= t and d < dt])
        correct = [get_count(t, distance_threshold) for t in thresholds]
        total = [get_count(t, 9999999) for t in thresholds]
        return { "correct" : correct, "total" : total }

    if metric == "ratio" :
        correct = [r for d,r in zip(distances, ratios) if d < distance_threshold]
        return { "correct" : correct, "total" : ratios }


def folderMatch(directory, match_fun, options) : 

    # Get distance_threshold
    distance_threshold      = options.get("distance_threshold", 5)

    def flatten(l) :
        return list(itertools.chain.from_iterable(l))

    def nb_correct_matches(distances) :
        return len([d for d in distances if d <= distance_threshold])

    # Get image pairs of folder
    imagePairs = getImagePairs(directory)

    # Get a set of matches varied over image pairs (matrix of size N x T where N
    # is amount of image pairs and T is the amount of thresholds. Each element in
    # the matrix is a zipped list of matches and distances
    def do_match(p, h) :
        print("%s " % features.getLabel(p[0])),
        return numpy.array(match_fun(p, options)[0])

    # Get matches
    match_sets = numpy.array([do_match(p, h) for p, h in imagePairs]).T

    # Now collect an x axis with nb correct matches per threshold
    get_correct_matches = lambda match_row : [nb_correct_matches(d) for d in match_row]
    nb_correct_set = [get_correct_matches(row) for row in match_sets]

    # Now collect an x axis with nb correct matches per threshold
    get_total = lambda match_row : [len(d) for d in match_row]
    nb_total_set = [get_total(row) for row in match_sets]

    return nb_correct_set, nb_total_set


def folderCorrespondences(directory, options) :

    # Get image pairs of folder
    imagePairs = getImagePairs(directory)

    # Calculate the number of correspondences
    correspondences = [getCorrespondences(p, h, options) for p, h in imagePairs]

    return correspondences


# Load all files
def getImagePairs(directory) :
    dir_path = "../../images/testsets/%s/" % directory
    pairs = [dir_path + p for p in os.listdir(dir_path) if len(p) <= 6]
    homographies = [getHomography(p) for p in pairs]
    return [([p + "_1.jpg", p + "_2.jpg"], h) for (p, h) in zip(pairs, homographies)]



def getHomography(hom_path) :
    h = numpy.loadtxt(hom_path)
    return h


def getCorrespondences(paths, homography, options = {}) :

    # Get options
    distance_threshold      = options.get("distance_threshold", 5)
    verbose                 = options.get("evaluate_verbose", False)

    # Get all feature points
    indices, ks, ds = features.getFeatures(paths, options)

    # Get all positions
    (pos_im1, pos_im2) = (features.getPositions(ks[indices == 0]), features.getPositions(ks[indices == 1]))

    # Translate positions in image2
    def im1_to_im2(p) :
        m1to2 = homography.dot(numpy.array([p[0], p[1], 1]))
        return m1to2[0:2]/m1to2[2]

    def im2_to_im1(p) :
        m2to1 = numpy.linalg.inv(homography).dot(numpy.array([p[0], p[1], 1]))
        return m2to1[0:2]/m2to1[2]

    # Make ball tree from im2 to im1
    dist_tree = trees.init(pos_im2, "ball")
    pos_im1_trans = [im1_to_im2(p) for p in pos_im1]

    # Find neighbors
    neighbors = dist_tree(pos_im1_trans, 1)
    distances = [matchDistance(pos_im1[i], pos_im2[idx[0]], homography) for i, (idx, d) in enumerate(neighbors)]
    correspondences = len([d for d in distances if d < distance_threshold])

    if verbose :
        path = lambda p : "/".join(p.split("/")[-2:])
        print("Found %i correspondences for images (%s, %s)" % (correspondences, path(paths[0]), path(paths[1])))

    return correspondences



def getTestsetPaths(collection, index = None) :
    pairs = getImagePairs(collection)
    if index == None : return random.choice(pairs)
    else : return pairs[index]



def getPaths(orig_id = None, compare_id = 2, img_type = "graf", folder = "inria") :

    # Dictionary to load the files
    file_endings = {
        "GRAFFITI6" : "ppm",
        "bark" : "ppm",
        "bikes" : "ppm",
        "boat" : "pgm",
        "graf" : "ppm",
        "leuven" : "ppm",
        "trees" : "ppm",
        "ubc" : "ppm",
        "wall" : "ppm"
    }

    # Get File ending
    img_ending = file_endings.get(img_type,"ppm")
    #img_ending = "png"

    # Get directory
    dir_path = "../../images/%s/%s/" % (folder, img_type)

    # Get id of first image in series
    i_first = int(fnmatch.filter(os.listdir(dir_path), "H[0-9]to[0-9]*")[0][1])
    
    # let i1 be original id if specified, else we'll set it to i_first
    i1 = i_first if orig_id == None else orig_id

    # Get image basename
    basename = fnmatch.filter(os.listdir(dir_path), "*[a-z]%s.%s" % (i1,img_ending))[0].split(str(i1))[0]
    i2 = compare_id
    p1 = "%s%s%i.%s" % (dir_path, basename, i1, img_ending)
    p2 = "%s%s%i.%s" % (dir_path, basename, i2, img_ending)
    paths = [p1, p2]

    def getH(i) :
        hom_name = filter(re.compile("H%ito%i(p|prec)?$" % (i_first, i)).match, os.listdir(dir_path))[0]
        hom_path = "%s%s" % (dir_path, hom_name)
        return getHomography(hom_path)

    # Load and parse homography
    if i_first == i1 : 
        hom = getH(i2)
    else :
        hom_1 = getH(i1)
        hom_2 = getH(i2)
        hom = hom_2.dot(numpy.linalg.inv(hom_1))

    return paths, hom



def getACRPaths(orig_id = None, compare_id = 2, img_type = "graf") :
    return getPaths(orig_id, compare_id, img_type, "acr")


def matchDistance(p1, p2, hom) :
    """ Given a homography matrix and two points this function calculates
        the geometric distance between the first point transformed by the
        homography matrix and the second point
    """
    m1to2 = hom.dot(numpy.array([p1[0], p1[1], 1]))
    m2to1 = numpy.linalg.inv(hom).dot(numpy.array([p2[0], p2[1], 1]))
    p2_n = m1to2[0:2]/m1to2[2]
    p1_n = m2to1[0:2]/m2to1[2]
    return numpy.linalg.norm(p2_n - p2) + numpy.linalg.norm(p1_n - p1)



def getDistinctMatches(matches, threshold = 5) :
    
    def distinctFrom(m, head) :
        a = numpy.linalg.norm(numpy.array(m[0]) - numpy.array(head[0]))
        b = numpy.linalg.norm(numpy.array(m[1]) - numpy.array(head[1]))
        return a > threshold or b > threshold
    
    if len(matches) < 1 : 
        return []
    head = matches[0]
    tail = [m for m in matches[1:] if distinctFrom(m, head)]
    return [head] + getDistinctMatches(tail, threshold)



def clusterMatch(distance_threshold, paths, homography, thresholds, keypoint, descriptor) :
    options = {
        "prune_fun" : weightMatrix.pruneThreshold, 
        "prune_limit" : 2.5,
        "min_coherence" : 0.0,
        "thresholds" : thresholds,
        "split_limit" : 50000,
        "cluster_prune_limit" : 1.5,
        "distance_threshold" : distance_threshold,
        "keypoint_type" : keypoint,
        "descriptor_type" : descriptor,
        "verbose" : False,
    }
    match_fun = clustermatch.match(paths, options)
    return evaluate(match_fun, thresholds, homography)



def spectralMatch(distance_threshold, paths, homography, thresholds, keypoint, descriptor) :
    options = {
        "thresholds" : thresholds,
        "distance_threshold" : distance_threshold,
        "keypoint_type" : keypoint,
        "descriptor_type" : descriptor,
        "match_fun" : ratiomatch.getMatchSet,
    }
    match_fun = spectralmatch.match(paths, options)
    return evaluate(match_fun, thresholds, homography)



def spectralMatchMMC(distance_threshold, paths, homography, thresholds, keypoint, descriptor) :
    options = {
        "distance_threshold" : distance_threshold,
        "keypoint_type" : keypoint,
        "descriptor_type" : descriptor,
        "match_fun" : clustermatch.getMatchSet,
        "threshold" : 0.94,
        "verbose" : False,
    }
    match_fun = spectralmatch.matchAlt(paths, options)
    return evaluate(match_fun, thresholds, homography)



def ratioMatch(distance_threshold, paths, homography, thresholds, keypoint, descriptor) :
    options = {
        "thresholds" : thresholds,
        "distance_threshold" : distance_threshold,
        "keypoint_type" : keypoint,
        "descriptor_type" : descriptor,
    }
    match_fun = ratiomatch.match(paths, options)
    return evaluate(match_fun, thresholds, homography)


def standardMatch(distance_threshold, paths, homography, thresholds, keypoint, descriptor, ratio) :
    options = {
        "thresholds" : thresholds,
        "distance_threshold" : distance_threshold,
        "keypoint_type" : keypoint,
        "descriptor_type" : descriptor,
        "ratio" : ratio,
        #"verbose" : True,
    }
    match_fun = mirrormatch.match(paths, options)
    return evaluate(match_fun, thresholds, homography)



def ballMatch(distance_threshold, paths, homography, thresholds, keypoint, descriptor) :
    options = {
       "leaf_size": 10,
       "radius_size": 250,
       "dist_threshold": 300,
       "ratio_boost" : 1.2,
       "shuffle_keypoints" : False,
       "group_limit" : 4,
       "match_type" : "radius",
    }
    match_fun = ballmatch.match(paths, options)
    return evaluate(match_fun, thresholds, homography)


def isoMatch(distance_threshold, paths, homography, thresholds, keypoint, descriptor) :
    options = {
        "thresholds" : thresholds,
        "distance_threshold" : distance_threshold,
        "keypoint_type" : keypoint,
        "descriptor_type" : descriptor,
    }
    match_fun = isomatch.match(paths, options)
    return evaluate(match_fun, thresholds, homography)
