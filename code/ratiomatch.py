"""
Python module for matching two images using matching from original sift paper

Based on:
@article{lowe2004sift,
  title={Distinctive image features from scale-invariant keypoints},
  author={Lowe, David G},
  journal={International journal of computer vision},
  volume={60},
  number={2},
  pages={91--110},
  year={2004},
  publisher={Springer}
}

Jonas Toft Arnfred, 2013-07-17
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

    keypoint_type		= options.get("keypoint_type", "SIFT")
    descriptor_type		= options.get("descriptor_type", "SIFT")

    # Get all feature points
    indices, ks, ds = features.getFeatures(paths, keypoint_type = keypoint_type, descriptor_type = descriptor_type)

    # Use cv2's matcher to get matching feature points
    ii, ss, uu = features.bfMatch(descriptor_type, ds[indices == 0], ds[indices == 1])
    
    # Get all positions
    (pos_im1, pos_im2) = (features.getPositions(ks[indices == 0]), features.getPositions(ks[indices == 1]))

    # Define a function that given a threshold returns a set of matches
    def match_fun(threshold) :
        match_data = [(numpy.array((pos_im1[i], pos_im2[j])), uu[i], ss[i]) for i,j in enumerate(ii) if uu[i] < threshold]
        if len(match_data) == 0 : return [], [], []
        matches, ratios, scores = zip(*match_data)

        return matches, ratios, scores

    return lambda t : match_fun(t)


def getMatchSet(paths, options = {}) :

    # Get parameters
    threshold = options.get("threshold", 1.0)

    match_fun = match(paths, options)

    return match_fun(threshold)
