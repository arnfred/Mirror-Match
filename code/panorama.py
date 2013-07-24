"""
Python module for stitching together panorama images based on the MMC algorithm

Jonas Toft Arnfred, 2013-03-08
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import os
import fnmatch
import features
import display
import louvain
import weightMatrix
import clustermatch
import numpy
import cv2
import graph_tool.all as gt
import pylab
from string import ascii_uppercase, digits
from os.path import isdir, dirname, exists
import random
from itertools import combinations, groupby, product
from scipy.interpolate import griddata
from scipy.misc import imresize



####################################
#                                  #
#           Functions              #
#                                  #
####################################

def cropAndResize(images) : 
    """ Crop and resize images to make the less expensive to work with """

    for image in images :
        #print(image)
        arg1 = "convert \"%s\" -resize 40%% \"%s\"" % (image, image)
        arg2 = "convert \"%s\" -gravity center -crop 80%%x90%%+0+0 +repage \"%s\"" % (image, image)
        os.popen(arg1)
        os.popen(arg2)



def getPartitions(paths, options = {}) :
    """ Returns a partitioning of features 
        Copy pasta from clustermatch
    """

    # Get parameters
    prune_limit = options.get("prune_limit", 2.5)
    keypoint_type = options.get("keypoint_type", "SIFT")
    descriptor_type = options.get("descriptor_type", "SIFT")
    verbose = options.get("verbose", False)
    split_limit = options.get("split_limit", 50)
    cluster_prune_limit = options.get("cluster_prune_limit", 1.5)

    # Get all feature points
    indices, ks, ds = features.getFeatures(paths, keypoint_type = keypoint_type, descriptor_type = descriptor_type)

    # Calculate weight matrix
    weights = weightMatrix.init(ds, descriptor_type)

    # Get cluster weights
    cluster_weights = weightMatrix.pruneThreshold(weights, prune_limit)

    # Cluster graph
    partitions = clustermatch.cluster(cluster_weights, indices, split_limit = split_limit, prune_limit = cluster_prune_limit, verbose=verbose)
    if verbose : print("%i partitions" % len(set(partitions)))

    return indices, ks, lambda t : list(clustermatch.getPartitionMatches(partitions, cluster_weights, weights, indices, t))



# Get positions
def getPos(indices, ks, match_data) :
    positions = features.getPositions(ks)
    pos = [(positions[m1], positions[m2]) for ((m1,m2), u, s) in match_data]
    img = [(indices[m1], indices[m2]) for ((m1,m2), u, s) in match_data]
    return pos, img



def getHomography(matches) :

    match_1, match_2 = zip(*matches)
    homography, blup = cv2.findHomography(numpy.array(match_1), numpy.array(match_2), cv2.RANSAC)

    return homography



def filterPositions(pair, pos, img) :
    return [pos for (im_pair, pos) in zip(img, pos) if im_pair == pair]



def occurences(ls) :
    tags = list(set(ls))
    count = [len(filter(lambda i : t == i, ls)) for t in tags]
    return [(c, t) for c, t in sorted(zip(count, tags))[::-1] if c >= 4]



def initGraph(edges, homographies, images, paths) :
    g = gt.Graph()
    g.set_directed(True)
    
    # initializing property maps
    g.ep["homographies"] = g.new_edge_property("python::object")
    g.ep["counts"] = g.new_edge_property("int")
    g.vp["images"] = g.new_vertex_property("python::object")
    g.vp["paths"] = g.new_vertex_property("string")
    
    vs = g.add_vertex(len(paths))
    ind = g.vertex_index
    img_set = set(zip(*edges)[1])
    
    # Add edges
    for (v1,v2) in combinations(vs, 2) :
        if (ind[v1], ind[v2]) in img_set : 
            e = g.add_edge(v1,v2)
            e = g.add_edge(v2,v1)
    
    # Add homographies
    for (count, (i1, i2)), h in zip(edges, homographies) :
        v1, v2 = g.vertex(i1), g.vertex(i2)
        e = g.edge(v1,v2)
        e_inv = g.edge(v2,v1)
        h_inv = numpy.linalg.inv(h)
        h_inv = h_inv / h_inv[2,2] # Normalize
        g.ep["homographies"][e] = h
        g.ep["homographies"][e_inv] = h_inv
        g.ep["counts"][e] = count
        g.ep["counts"][e_inv] = count
    
    # Add images and their paths
    for (i,(im,p)) in enumerate(zip(images, paths)) :
        g.vp["images"][g.vertex(i)] = im
        g.vp["paths"][g.vertex(i)] = p
        
    return g



def getRandPath(size=6, chars=ascii_uppercase + digits) :
    condition = True
    while condition:
        name = ''.join(random.choice(chars) for x in range(size))
        condition = exists(name)
    return name



def combine(im1, im2, h_1_2, scale = 0.5) :
    """ Combining two images """
    
    # Utility function for calculating the transformation of a point with homography
    def fromHom(p,h) :
        crd = h.dot(numpy.array([p[0], p[1], 1]))
        crd_norm = crd / crd[2]
        return (crd_norm[0], crd_norm[1])

    # Get image coordinates
    (im1_y, im1_x) = im1.shape
    (im2_y, im2_x) = im2.shape

    print(h_1_2)
    
    # Collect data points for the transformed image
    im_data = numpy.zeros((im1.shape[0]*im1.shape[1],3))
    for k,(x,y) in enumerate(product(range(0, im1_x), range(0, im1_y))) :
        (x_new, y_new) = fromHom((x,y),h_1_2)
        im_data[k,:] = numpy.array((x_new, y_new, im1[y,x]))

    # Get max and min values
    max_x = min(numpy.max(im_data[:,0])+1, 1000)
    max_y = min(numpy.max(im_data[:,1])+1, 1000)
    min_x = max(numpy.min(im_data[:,0]), -1000)
    min_y = max(numpy.min(im_data[:,1]), -1000)

    # Calculate padding for canvas
    padding_min_x = max(min_x,0)
    padding_max_x = max(im2_x - max_x, 0)
    padding_min_y = max(min_y,0)
    padding_max_y = max(im2_y - max_y, 0)

    # Interpolate result
    print("max: (%i,%i), min: (%i,%i)" % (max_x, max_y, min_x, min_y))
    print("Creating a grid of side: %i x %i" % (max_y+padding_max_y - min_y+padding_min_y, max_x+padding_max_x - min_x+padding_min_x))
    grid_y, grid_x = numpy.mgrid[min_y-padding_min_y:max_y+padding_max_y, min_x-padding_min_x:max_x+padding_max_x]
    im_interp = griddata(im_data[:,0:2], im_data[:,2], (grid_x, grid_y), method='linear', fill_value=0)

    # Cut highlights
    im_interp[im_interp > 255] = 255
    im_interp[im_interp < 0] = 0

    # Fill in im2
    offset_x = max(-1*min_x,0)
    offset_y = max(-1*min_y,0)
    for x, y in product(range(0, im2_x), range(0, im2_y)) :
        if im2[y,x] > 0 :
            im_interp[y + offset_y, x + offset_x] = im2[y,x]

    # Scale result
    im_r = imresize(im_interp, size=float(scale), interp='bicubic')

    # Debug output
    path = "%s.jpg" % getRandPath()
    #imshow(im_r)
    pylab.imsave(path, im_r)

    # Scaling homography
    h_s = numpy.identity(3)
    h_s[0,0] = scale
    h_s[1,1] = scale

    # Now calculate new_h1, new_h2
    h_1_r = numpy.identity(3)
    h_1_r[0,2] = -1*min_x
    h_1_r[1,2] = -1*min_y
    h_1_r = h_s.dot(h_1_r).dot(h_1_2)

    h_2_r = numpy.identity(3)
    h_2_r[0,2] = max(-1*min_x,0)
    h_2_r[1,2] = max(-1*min_y,0)
    h_2_r = h_s.dot(h_2_r)

    return im_r, h_1_r, h_2_r, path



def step(graph) :

    def getInv(h) :
        h_inv = numpy.linalg.inv(h)
        h_inv = h_inv / h_inv[2,2]
        return h_inv

    counts = graph.ep["counts"]
    images = graph.vp["images"]
    homographies = graph.ep["homographies"]
    paths = graph.vp["paths"]

    # Find edge with highest weight
    max_edge = None
    for e in graph.edges() :
        if max_edge == None or counts[e] > counts[max_edge] :
            max_edge = e

    # Get data about this edge
    (v1,v2) = max_edge
    im1 = images[v1]
    im2 = images[v2]
    h_1_2 = homographies[max_edge]

    # Combine the images of the two vertices linked by the edge
    r, h_1_r, h_2_r, r_path = combine(im1, im2, h_1_2)

    # Calculate inverse homographies and identity
    h_r_1 = getInv(h_1_r)
    h_r_2 = getInv(h_2_r)
    h_r_r = numpy.identity(3)

    # For all edges leading from and to v1 and v2, recalculate homographies
    for e in v1.in_edges() :
        h_k_1 = homographies[e]
        h_k_r = h_1_r.dot(h_k_1)
        homographies[e] = h_k_r

    for e in v1.out_edges() :
        h_1_k = homographies[e]
        h_r_k = h_1_k.dot(h_r_1)
        homographies[e] = h_r_k

    for e in v2.in_edges() :
        h_k_2 = homographies[e]
        h_k_r = h_2_r.dot(h_k_2)
        homographies[e] = h_k_r

    for e in v2.out_edges() :
        h_2_k = homographies[e]
        h_r_k = h_2_k.dot(h_r_2)
        homographies[e] = h_r_k

    # Update image for v1
    images[v1] = r
    paths[v1] = r_path

    # Now move all edges leading to v2 over to v1
    print("v1 has %i edges" % len(list(v1.all_edges())))
    print("v2 has %i edges" % len(list(v2.all_edges())))
    for (vi, _) in v2.in_edges() :
        if vi != v1 :
            print("moving edge between %i and %i so it becomes an edge between %i and %i" % (v2, vi, v1, vi))
            e_in = graph.add_edge(vi, v1)
            e_out = graph.add_edge(v1, vi)
            h_in = graph.edge(vi,v2)
            h_out = graph.edge(v2,vi)
            homographies[e_in] = homographies[h_in]
            homographies[e_out] = homographies[h_out]
            counts[e_in] = counts[h_in]
            counts[e_out] = counts[h_out]

    print("Contracting vertex: %i and %i" % (v1, v2))
    # remove v2
    graph.clear_vertex(v2)
    graph.remove_vertex(v2)



def drawGraph(graph, filename = "panorama_graph.png") :

    # Calculate weights
    weights = graph.new_edge_property("double")
    for e in graph.edges() :
        weights[e] = pylab.log(graph.ep["counts"][e]) + 1

    pos = gt.sfdp_layout(graph, eweight=weights)
    gt.graph_draw(graph, 
              pos=pos, 
              output_size=(2000, 2000), 
              #vertex_halo=True, 
              #vertex_fill_color=partitioning, 
              vertex_anchor=0,
              vertex_pen_width=10,
              vertex_surface=graph.vp["paths"],
              vertex_size=300, 
              edge_pen_width=weights, 
              output=filename)
    im = pylab.imread(filename)
    pylab.imshow(im)


