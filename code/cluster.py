"""
Python module for use with OpenCV to cluster descriptors from several images.

Jonas Toft Arnfred, 2013-03-19
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import cv2
import pylab
import numpy
import display
import features
import math
from itertools import combinations, groupby, tee, product, combinations_with_replacement, dropwhile
import graph_tool.all as gt



####################################
#                                  #
#           Properties             #
#                                  #
####################################

feature_keypoint = "ORB"
feature_descriptor = "BRIEF"
N = 600


####################################
#                                  #
#           Functions              #
#                                  #
####################################



def cluster(graph, nb_clusters=20, iter=100, gamma=1.0, corr="erdos", verbose=False) :

	# Get weights (throws an error if they don't exist)
	weights = graph.edge_properties["weights"]

	clusters = gt.community_structure(graph, iter, nb_clusters, weight=weights, gamma=gamma, corr=corr, verbose=verbose)
	t = [(k, len(list(g))) for k,g in groupby(sorted(clusters.fa))]
	return clusters, len(t)



def scoreImagePair(paths, clustering=cluster) :
	""" Given paths to two images, the images are scored based on how 
	    well their traits match
	"""
	# Get features
	indices, keypoints, descriptors = getDescriptors(paths)

	# Construct graph
	connected, weights = initGraph(descriptors, indices)
	connected_light = prune(connected, weights, edges_per_vertex=20)
	filtered = prune(connected, weights, edges_per_vertex=5)

	# Cluster graph
	clusters, nb_clusters = clustering(filtered)

	# Match the traits
	scores, cluster_indices = scoreAll(connected_light, clusters, nb_clusters)

	# Get labels
	[l1, l2] = [features.getLabel(p) for p in paths]

	# Get score
	if len(scores) > 2 :
		score = sum(scores)
		print("Score: %0.4f for %s and %s" % (score,l1,l2))
	else :
		score = -0.1
		print("Score: %0.4f for %s and %s (Less than two matches)" % (score,l1,l2))

	return score



def getDescriptors(paths) :
	""" Given a list of paths to images, the function returns a list of 
	    descriptors and keypoints
		Input: paths [list of strings] The paths to the images we are using
		Out:   [pair of list of descriptors and list of keypoints]
	"""
	# Get all images and labels
	labels = map(features.getLabel, paths)
	images = map(features.loadImage, paths)

	# Get feature descriptors
	#keypoints = [features.getKeypoints(feature_keypoint, im) for im in images]
	keypoints = [features.getORBKeypoints(im) for im in images]
	data = [features.getDescriptors(feature_descriptor, im, k) for (im, k) in zip(images, keypoints)]
	keypoints, descriptors = zip(*data)

	indices = [l for i,n in zip(range(len(labels)), map(len, descriptors)) for l in [i]*n]
	return (indices, numpy.concatenate(keypoints), numpy.concatenate(descriptors))



def setPositions(graph, keypoints) :

	positions = display.getPositions(keypoints)

	posMap = graph.new_vertex_property("vector<float>")
	for v in graph.vertices() : posMap[v] = positions[graph.vertex_index[v]]

	# Save weights as a graph property
	graph.vertex_properties["positions"] = posMap

	return graph



def reassign(partitioning) :
    assignments = [(i,k) for i,(k,g) in enumerate(groupby(sorted(partitioning.fa)))]
    p_new = partitioning.copy()
    for i,k in assignments :
        p_new.fa[partitioning.fa == k] = i
    return p_new



def initGraph(descriptors, indices, dampener = 1.0) :

	# This is a faster way to initialize a graph. It's not random.
	N = len(descriptors)
	graph = gt.random_graph(N, lambda: N - 1, directed=False, random=False)

	# Because the random graph is generated with scrambled indices we need to reorder
	graph.reindex_edges()

	# Create indicemap
	ind = graph.new_vertex_property("int")
	ind.fa = indices
	graph.vertex_properties["indices"] = ind

	# Define class colors
	class_colors = graph.new_vertex_property("vector<float>")
	colors = [[0.1, 0.1, 0.1, 0.9], [0.7, 0.7, 0.7, 0.9], [0.3, 0.3, 0.3, 0.9], [0.9, 0.9, 0.9, 0.9], [0.5, 0.5, 0.5, 0.9],
			  [0.2, 0.2, 0.2, 0.9], [0.8, 0.8, 0.8, 0.9], [0.4, 0.4, 0.4, 0.9], [0.6, 0.6, 0.6, 0.9], [1.0, 1.0, 1.0, 0.9]] 
	for v in graph.vertices() : class_colors[v] = colors[ind[v]]
	graph.vertex_properties["class_colors"] = class_colors

	# Set weights
	distances = hammingDist(descriptors)
	weights = setWeights(graph, distances, dampener)

	# Filter all zero-edges
	weight_prop = graph.edge_properties["weights"]
	non_zero = graph.new_edge_property("bool")
	non_zero.fa = (weight_prop.fa != 0)

	# Then construct a graphView reflecting this
	filtered = gt.GraphView(graph, efilt=non_zero)

	return filtered, weights



def setWeights(graph, distances, dampener = 1.0) :

	# Normalize hamming distances
	max_d = numpy.max(distances)
	distances_normalized = numpy.array(distances, dtype=numpy.float) / numpy.array(max_d, dtype=numpy.float)
	weights = 1 - distances_normalized

	# Find some stats about the weights
	u = numpy.mean(weights)
	sd = numpy.sqrt(numpy.var(weights))

	# Set weights that are too high to 0
	weights[(weights == 1)] = 0

	# Boost inter-image edges
	dampenIntraImageEdges(graph, weights, dampener)

	# Update degree
	setDegree(graph, weights)

	# Set weights
	weight_prop = graph.new_edge_property("float")
	weight_prop.fa = [(weights[graph.vertex_index[e.target()], 
							 graph.vertex_index[e.source()]]) for e in graph.edges()]

	# Store weights as graph property
	graph.edge_properties["weights"] = weight_prop

	return weights



def dampenIntraImageEdges(graph, weights, dampener = 1.0) :
	
	indices = graph.vp["indices"]
	nb_keypoints_im0 = numpy.sum(indices.fa - 1)
	length = weights.shape[0]

	# Set the intra-image edges to zero
	mask = numpy.zeros(weights.shape, dtype=numpy.bool)
	mask[0:(nb_keypoints_im0-1), 0:(nb_keypoints_im0-1)] = True
	mask[nb_keypoints_im0:(length-1), nb_keypoints_im0:(length-1)] = True

	# Multiply the inter image edges with the boost factor
	weights[mask] = weights[mask] / dampener


def setDegree(graph, weights) :
	degrees = numpy.array(map(numpy.sum, weights))
	g_degrees = graph.new_vertex_property("float")
	g_degrees.fa = degrees
	graph.vp["degrees"] = g_degrees



def removeIntraImageEdges(graph, weights) : 
	# Filter edges within images
	indices = graph.vertex_properties["indices"]
	inter_edges = graph.new_edge_property("bool")

	#Get amount of points for each image
	nb_keypoints_im1 = numpy.sum(indices.fa)
	length = weights.shape[0]

	# Set the intra-image edges to zero
	weights[0:(nb_keypoints_im1-1), 0:(nb_keypoints_im1-1)] = 0
	weights[nb_keypoints_im1:(length-1), nb_keypoints_im1:(length-1)] = 0

	# Update degree
	setDegree(graph, weights)

	# Create edge_Filter
	for e in graph.edges() : inter_edges[e] = (indices[e.source()] != indices[e.target()])

	return gt.GraphView(graph, efilt=inter_edges)



def get_treshold(weights, edges_per_vertex) :
	n = 200
	nb_vertices = weights.shape[0]
	tresholds = numpy.linspace(1,0.5,n)
	w = [numpy.sum(weights > t) / (2.0*nb_vertices) for t in tresholds]
	q = dropwhile(lambda t : t < edges_per_vertex, w)
	index = len(list(q))
	return tresholds[n-index]


def prune(graph, weights, edges_per_vertex) :
	""" Removes all edges under a certain treshold
	"""
	# Get treshold and weights
	treshold = get_treshold(weights, edges_per_vertex)
	g_weights = graph.ep["weights"]

	# Set weights below a treshold to zero
	index = (g_weights.fa > treshold) & (g_weights.fa < 1)

	# Filter all zero-edges
	g_treshold = graph.new_edge_property("bool")
	g_treshold.fa = index

	# Then construct a graphView reflecting this
	filtered = gt.GraphView(graph, efilt=g_treshold)

	# Update degrees
	index = (weights <= treshold) | (weights == 1)
	weights[index] = 0
	setDegree(filtered, weights)

	return filtered



def modularity(graph, f) :
	""" Calculates the modularity of the cluster marked by 'f' in the graph
	    Input: Graph [Graph] The parent graph of the cluster
	           f [boolean Graph vertex property] A filter marking the vertices 
	           that form the cluster
	"""
	# Function for getting all edges of vertex
	def sumEdges(vertex) : return numpy.sum([weights[e] for e in vertex.all_edges()])

	# create partition graph and get weights
	p = gt.GraphView(graph, vfilt=f)
	weights = graph.edge_properties["weights"]

	# Calculate modularity
	two_m = 2*numpy.sum([weights[e] for e in graph.edges()])
	#two_m = numpy.sum([sumEdges(v) for v in graph.vertices()])
	internal_sum = sum([sumEdges(v) for v in p.vertices()])
	external_sum = sum([sumEdges(v) for v in graph.vertices() if f[v]])
	fraction = (internal_sum / (two_m))
	E_fraction = (external_sum / (two_m)) ** 2
	return fraction - E_fraction



def coherence(graph, f) : -1 * modularity(graph, f)



def getFilter(graph, partitioning, indices) :
	""" Given a partitioning and some indices, return a boolean vertex property
	    marking the vertices as true if they belong to the partitions
	    input: graph [Graph]
	           partitioning: [Integer Vertex property] A partitioning of the graph
	           indices: An integer or a list of integers denoting which clusters to use
	    output: Boolean vertex property
	"""
	if isinstance(indices, ( int, long ) ) : indices = [indices]
	f_new = numpy.zeros(partitioning.fa.shape, dtype=numpy.bool)
	for i in indices :
		f_new = (partitioning.fa == i) | f_new

	# Create a filter
	f = graph.new_vertex_property("bool")
	f.fa = f_new
	return f



def andFilter(graph, f, partitioning, indices) :
	""" Given a filter, a partitioning and some indices, return a boolean vertex property
	    marking the vertices as true if they belong to the partitions and to the previous filter
	    input: graph [Graph]
	           f [Boolean vertex property] The filter we are extending
	           partitioning: [Integer Vertex property] A partitioning of the graph
	           indices: An integer or a list of integers denoting which clusters to use
	    output: Boolean vertex property
	"""
	if isinstance(indices, ( int, long ) ) : indices = [indices]
	f_new = numpy.zeros(partitioning.fa.shape, dtype=numpy.bool)
	for i in indices :
		f_new = (partitioning.fa == i) | f_new

	# Create a filter
	f_ret = graph.new_vertex_property("bool")
	f_ret.fa = f_new & numpy.array(f.fa, dtype=numpy.bool)
	return f_ret



def getPartition(graph, partitioning, partition_indices) :
	""" Returns a partition corresponding to the partitioning and the indices
	"""
	# Make sure indices are a list
	if isinstance(partition_indices, ( int, long ) ) : partition_indices = [partition_indices]

	# Add all qualified vertices to filter
	f = numpy.zeros(partitioning.fa.shape, dtype=numpy.bool)
	for i in partition_indices :
		f = (partitioning.fa == i) | f
		
	# Create new filter
	new_f = graph.new_vertex_property("bool")
	new_f.fa = f

	# Create graph_view for the cluster
	return gt.GraphView(graph, vfilt=new_f)



def scoreClusterPair(graph, clusters, cluster, indices, i,j, weight=1.0, verbose=False) : 
	""" calculates the similarity score for a set of two images
	    input:  graph [Graph] the parent graph
	            clusters [int vertex property] partitioning of graph corresponding to clusters
	            indices [int vertex property] partitioning of graph corresponding to images
	            i [int] index of first image
	            j [int] index of second image
	            weight [double] The weight for this particular cluster
	"""
	def u_s(m,c) : return (m + 3*c)
	def s(m,c,w) : return u_s(m,c) * w
	def printResult(m, c, w, i, j) : 
		print("[%i]" % cluster + "\tm: %+0.4f" % m + ", c: %+0.4f" % c + ", u_s: %+0.4f" % u_s(m,c) + ", s: %+0.4f" % (s(m,c,w)) + ", w: %0.2f" % w)

	f_c = getFilter(graph, clusters, cluster)
	f_ij = andFilter(graph, f_c, indices, [i,j])
	f_i = andFilter(graph, f_c, indices, [i])
	f_j = andFilter(graph, f_c, indices, [j])

	graph_ij = gt.GraphView(graph, vfilt=f_ij)
	graph_c = gt.GraphView(graph, vfilt=f_c)

	m = modularity(graph, f_ij)
	c = -1 * (modularity(graph_ij, f_i) + modularity(graph_ij, f_j))
	if verbose : printResult(m,c,weight,i,j)
	return s(m,c,weight)



def scoreAll(graph, clusters, nb_clusters, verbose=False) :
    # Define range and precalculate filters
	n = nb_clusters - 1 # amount of clusters
	filters = [getFilter(graph, clusters, c) for c in range(n)]
	mods = [modularity(graph, filters[c]) for c in range(n)]
	indices = graph.vertex_properties["indices"]

	def in_range(c) :
		ones = (numpy.sum(indices.fa*filters[c].fa) > 0)
		zeros = (numpy.sum((1-indices.fa)*filters[c].fa) > 0)
		return ones and zeros
	
	def score(c) : 
		return scoreClusterPair(graph, clusters, c, indices, 0, 1, mods[c]/mods_sum, verbose=verbose)

	cluster_indices = [c for c in range(n) if in_range(c)]
	mods_sum = sum([mods[c] for c in cluster_indices])

	# Get scores and filter them for NaN
	scores = [score(c) for c in cluster_indices]
	filtered_scores = map(lambda s : 0 if math.isnan(s) else s, scores)

	return scores, cluster_indices



def hammingDist(descriptors) :
	""" Returns a matrix of size n x n where n is the amount of descriptors
		output[0][1] is equal to the hamming distance between descriptors[0] and descriptors[1]
		where output is the matrix returned from this function
	"""
	# rearrange inputs
	binary = numpy.array(numpy.unpackbits(descriptors), dtype=numpy.bool)
	binary.shape = (descriptors.shape[0], descriptors.shape[1] * 8)

	# Initialize return matrix
	result = numpy.zeros([descriptors.shape[0], descriptors.shape[0]], dtype=numpy.uint8)

	# Fill result matrix and return
	for (i, bin_row) in enumerate(binary) :
		result[i] = numpy.sum(numpy.bitwise_xor(binary, bin_row), 1)
	return result
