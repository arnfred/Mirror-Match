"""
Python module to provide a wrapper for a few tree types
"""
from sklearn.neighbors.ball_tree import BallTree
import pyflann

# Ball tree
def init(data, tree_type, options = {}) :
    if tree_type == "ball" :
        return ball_init(data, options)
    elif tree_type == "flann" :
        return flann_init(data, options)


def ball_init(data, options) :

    # Get options
    leaf_size           = options.get("leaf_size", 10) 
    dist_metric         = options.get("dist_metric", "minkowski") 
    verbose             = options.get("verbose", False)
    
    # Construct main ball tree
    tree = BallTree(data, leaf_size=leaf_size, metric=dist_metric)

    def query(D, k) :
        if verbose : print("."),
        result = [tree.query(d, k = k) for d in D]
        return [(i[0], d[0]) for d, i in result]

    return query


def flann_init(data, options) :

    # Get options
    precision           = options.get("precision", 0.9) 
    verbose             = options.get("verbose", False)

    # Construct main flann tree
    flann = pyflann.FLANN()
    params = flann.build_index(data, algorithm="autotuned", target_precision=precision)#, log_level = "info")

    def query(D, k) :
        if verbose : print("."),
        idx, dist = flann.nn_index(D, k, checks=params["checks"]);
        return [(i, d) for i, d in zip(idx, dist)]

    return query
