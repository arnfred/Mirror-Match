"""
Python module for matching two images using clustering with ball tress

Jonas Toft Arnfred, 2013-08-28

"""
####################################
#                                  #
#            Imports               #
#                                  #
####################################

import features
import numpy
from sklearn.neighbors.ball_tree import BallTree


####################################
#                                  #
#           Functions              #
#                                  #
####################################

def match(paths, options = {}) :

    keypoint_type		= options.get("keypoint_type", "SIFT")
    descriptor_type		= options.get("descriptor_type", "SIFT")
    leaf_size		    = options.get("leaf_size", 2)
    radius_size         = options.get("radius_size", 300)
    dist_threshold      = options.get("dist_threshold", 100)

    # Get all feature points
    indices, ks, ds = features.getFeatures(paths, keypoint_type = keypoint_type, descriptor_type = descriptor_type)

    # Construct ball tree
    bt = BallTree(ds, leaf_size=leaf_size)

    # Filter nodes
    ns = filterNodes(bt, radius_size)

    def match_fun(ratio_threshold) :
        return zip(*list(getMatches(bt, ns, indices, ks, ds, dist_threshold, ratio_threshold)))

    return match_fun



# utility function for getting a node
def getNode(bt, index) :
    return bt.node_data[index]


# Produce new ball tree with node as head.  This could be done more
# efficiently, but that would be tricky because of the underlying data
# structure of the ball tree
def newTree(bt, ds, index) :
    node = getNode(bt, index)
    idx = numpy.array(bt.idx_array[node['idx_start']: node['idx_end']+1])
    node_features = ds[idx]
    node_bt = BallTree(node_features, leaf_size=5)
    return (node_bt, idx)


# I'd like to go through the tree and find all the nodes first nodes (counting
# from the top) that have a radius less than a certain size
def filterNodes(bt, radius) :
    # Initialize an empty set of nodes
    akin = []
    
    # Initialize a list of nodes to skip
    skip = numpy.zeros(bt.node_data.shape, dtype=numpy.bool)
    
    # For each node, add it to akin if the node is small enough and shouldn't be skipped
    for i,n in enumerate(bt.node_data) :
        if n['radius'] < radius and not skip[i] :
            
            # Update skip list so children aren't expanded
            # If parent is in pos 'i', then the two children of the binary tree are at pos '2*i + 1' and '2*i+2'
            j = 2*i+1
            k = 2
            while j < bt.node_data.shape[0] :
                #print("skipping %i:%i" % (j, j-1+k))
                skip[j : j + k] = True
                k = k*2
                j = 2*j+1
            
            # Add node to akin
            akin.append(i)
    
    # Return akin
    return akin


# Nearest neighbor of a ball tree
def NN(descriptor, tree) :
    (dist, index) = tree.query(descriptor, k=3)
    return index[0][1], dist[0][1], dist[0][1]/float(dist[0][2])


# The criteria for keeping a match
def keepMatch(m, n, s, u, nn, indices, idx, dist_threshold, ratio_threshold) :
    keep = (s < dist_threshold                  # Should be fairly similar
        and u < ratio_threshold                 # Should be unique
        and indices[idx[m]] < indices[idx[n]])  # Should not be from the same image
        #and m == nn[n])                        # Should be both ways
    return keep


def getPos(m, n, indices, ks, idx) :
    if indices[idx[m]] < indices[idx[n]] :
        return features.getPositions([ks[idx[m]], ks[idx[n]]])
    else :
        return features.getPositions([ks[idx[n]], ks[idx[m]]])


def getMatches(bt, node_indices, indices, ks, ds, dist_threshold, ratio_threshold) :

    for node_index in node_indices :

        # Construct node tree
        node_tree, idx = newTree(bt, ds, node_index)

        # Get nearest neighbors
        nn, ss, uu = zip(*[NN(d, node_tree) for d in ds[idx]])
        #matches = [NN(d) for d in ds]

        # Yield match
        for m,n in enumerate(nn) :
            if keepMatch(m, n, ss[m], uu[m], nn, indices, idx, dist_threshold, ratio_threshold) :
                yield (numpy.array(getPos(m, n, indices, ks, idx)), ss[m], uu[m])
