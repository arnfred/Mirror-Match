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
import numpy


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
    verbose             = options.get("match_verbose", False)
    filter_features     = options.get("filter_features", [])

    # Get all feature points
    indices, ks, ds = getFeatures(paths, filter_features, options)

    if verbose : print("\n%i Keypoints found" % (len(ks)))

    # Check that we have enough features to continue
    if len(ks[indices == 0]) <= 3 :
        def fun(t) :
            return [], [], [], [], []
        return fun

    # Match
    match_data = features.match(ds, indices, options)

    if verbose : print("\n%i Matches found" % (len(match_data)))

    # Find all matches
    def get_matches() :
        seen = set([])

        # filter matches that aren't across images
        for (i, j), s, u in match_data :
            if (not (i, j) in seen) and indices[i] != indices[j] :
                seen.add((i, j))
                seen.add((j, i))
                if indices[i] < indices[j] : 
                    yield (i, j), s, u
                else :
                    yield (j, i), s, u


    # # Find all matches (Bothways)
    # def get_matches() :
    #     seen = set([])
    #     ways = set([])
    #     m = set([])
    #     for (i, j), s, u in match_data :
    #         if u < 1.0 :
    #             ways.add((i, j))
    #             m.add(((i, j), s, u))

    #     # filter matches that aren't both ways or are within only one image
    #     for (i, j), s, u in m :
    #         #if (j, i) in ways and (not (i, j) in seen) and indices[i] != indices[j] :
    #         if (not (i, j) in seen) and indices[i] != indices[j] :
    #         #if indices[i] != indices[j] :
    #             seen.add((i, j))
    #             seen.add((j, i))
    #             yield (i, j), s, u


    # Collect all matches
    matches = list(get_matches())

    # Define a function that given a threshold returns a set of matches
    def match_fun(threshold) :
        match_data = [(matchPos(i, j), s, u, (indices[i], indices[j]), (i,j)) for (i, j),s,u in matches if u < threshold]
        if len(match_data) == 0 : return [], [], [], [], []
        return zip(*match_data)

    return match_fun



def getFeatures(paths, filter_features, options) :
    """ Retrieves features and filters them """
    feature_points = features.getFeatures(paths, options)
    if filter_features == [] :
        return feature_points
    else :
        ff = numpy.ones(len(feature_points[0]), dtype = numpy.bool)
        ff[filter_features] = False
        indices = feature_points[0][ff]
        ks = feature_points[1][ff]
        ds = feature_points[2][ff]
        return indices, ks, ds
