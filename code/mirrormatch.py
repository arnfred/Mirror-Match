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
    with_pruned         = options.get("with_pruned", False)
    only_mirror         = options.get("only_mirror", False)

    # Add options
    options['group'] = only_mirror
    options['match_same'] = True

    # Get all feature points
    indices, ks, ds = features.getFeatures(paths, options)

    # Use cv2's matcher to get matching feature points
    if only_mirror :
        jj, ss, uu = features.mirrorMatch(ds, options)
    else :
        jj, ss, uu = features.bfMatch(ds, ds, options)

    # See if matches go both ways
    nb_images = len(paths)
    seen = set([])
    def add_match(i,j) :
        seen.add((i,j))
        seen.add((j,i))
        return (i,j)
    bothways = [add_match(i,j) for i, j in enumerate(jj) if (not (i,j) in seen) and (nb_images > 2 or jj[j] == i)]

    # Define a function that given a threshold returns a set of matches
    def match_fun(threshold) :
        match_data = [(matchPos(i, j), ss[i], uu[i], (indices[i],indices[j])) for i,j in bothways if uu[i] < threshold]
        if len(match_data) == 0 : return [], [], []
        return zip(*match_data)

    return match_fun


# C/P version for creating a figure containing the wrong match within an image
def match_show(paths, options = {}) : 
    # Returns the position of a match
    def matchPos(i,j) :
        return (features.getPosition(ks[i]), features.getPosition(ks[j]))

    # Get options
    with_pruned         = options.get("with_pruned", False)
    only_mirror         = options.get("only_mirror", False)

    # Add options
    options['group'] = only_mirror
    options['match_same'] = True

    # Get all feature points
    indices, ks, ds = features.getFeatures(paths, options)

    # Use cv2's matcher to get matching feature points
    if only_mirror :
        jj, ss, uu = features.mirrorMatch(ds, options)
    else :
        jj, ss, uu = features.bfMatch(ds, ds, options)

    # See if matches go both ways
    if not only_mirror : bothways = [(i, j) for i,j in enumerate(jj) if indices[i] == 0 and indices[j] == 1 and jj[j] == i]

    # Should we save the matches within one image?
    im1_ways = [(i, j) for i,j in enumerate(jj) if indices[i] == 0 and indices[j] == 0 and jj[j] == i]
    im2_ways = [(i, j) for i,j in enumerate(jj) if indices[i] == 1 and indices[j] == 1 and jj[j] == i]


    # Define a function that given a threshold returns a set of matches
    def match_fun(threshold) :
        match_data = [(matchPos(i,j), uu[i], ss[i]) for i,j in bothways if uu[i] < threshold]
        match_im1_data = [(matchPos(i,j), uu[i], ss[i]) for i,j in im1_ways if uu[i] < threshold]
        match_im2_data = [(matchPos(i,j), uu[i], ss[i]) for i,j in im2_ways if uu[i] < threshold]
        matches_both, ratios_both, scores_both = zip(*match_data)
        matches_im1, ratios_im1, scores_im1 = zip(*match_im1_data)
        matches_im2, ratios_im2, scores_im2 = zip(*match_im2_data)
        return matches_both, matches_im1, matches_im2

    return lambda t : match_fun(t)
