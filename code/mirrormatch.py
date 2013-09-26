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
    verbose             = options.get("verbose", False)

    # Add options
    options['match_same'] = True

    # Get all feature points
    indices, ks, ds = features.getFeatures(paths, options)

    # Use cv2's matcher to get matching feature points
    if only_mirror :
        match_data = features.mirrorMatch(ds, indices, options)
    else :
        match_data = features.bfMatch(ds, ds, options)

    if verbose : print("\n%i Matches found" % (len(match_data)))

    # Find all matches
    def get_matches() :
        seen = set([])
        nb_images = len(paths)
        for (i, j), s, u in match_data :
            if (not (i, j) in seen) and (nb_images > 2 or match_data[j][0][1] == i) :
                seen.add((i, j))
                seen.add((j, i))
                yield (i, j), s, u


    # Collect all matches
    matches = list(get_matches())

    # Define a function that given a threshold returns a set of matches
    def match_fun(threshold) :
        match_data = [(matchPos(i, j), s, u, (indices[i], indices[j])) for (i,j),s,u in matches if u < threshold]
        if len(match_data) == 0 : return [], [], [], []
        return zip(*match_data)

    return match_fun
