"""
Python module for use with openCV to cluster descriptors from several images.

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
from itertools import combinations
from itertools import groupby
import graph_tool.all as gt
from itertools import tee
from scipy.misc import imresize
import Image



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
	keypoints = [features.getKeypoints(feature_keypoint, im) for im in images]
	data = [features.getDescriptors(feature_descriptor, im, k) for (im, k) in zip(images, keypoints)]
	keypoints, descriptors = zip(*data)
	indices = [l for i,n in zip(range(len(labels)), map(len, descriptors)) for l in [i]*n]
	return (indices, numpy.concatenate(keypoints), numpy.concatenate(descriptors))



def initGraph(descriptors, indices) :

	# This is a faster way to initialize a graph. It's not random.
	N = len(descriptors)
	graph = gt.random_graph(N, lambda: N - 1, directed=False, random=False)

	# Because the random graph 
	graph.reindex_edges()

	# Set weights
	setWeights(graph, descriptors)

	# Create indicemap
	ind = graph.new_vertex_property("int")
	ind.fa = indices
	graph.vertex_properties["indices"] = ind

	return graph



def setPositions(graph, keypoints) :

	positions = display.getPositions(keypoints)

	posMap = graph.new_vertex_property("vector<float>")
	for v in graph.vertices() : posMap[v] = positions[graph.vertex_index[v]]

	# Save weights as a graph property
	graph.vertex_properties["positions"] = posMap

	return graph





def setWeights(graph, descriptors) :
	# Get all hamming distances
	distances = hammingDist(descriptors)

	# assign weights
	weights = graph.new_edge_property("int")
	weights.fa = [(distances[graph.vertex_index[e.target()], 
							 graph.vertex_index[e.source()]]) for e in graph.edges()]

	# Save weights as a graph property
	graph.edge_properties["weights"] = weights

	# Set normalized and inverted weights
	weights_normalized = graph.new_edge_property("float")
	norm_hamming = numpy.array(weights.a - numpy.min(weights.a), dtype=numpy.float) / numpy.array(numpy.max(weights.a) - numpy.min(weights.a), dtype=numpy.float)
	weights_normalized.fa = (1 - norm_hamming)

	# Store normalized weights as graph property
	graph.edge_properties["weights_normalized"] = weights_normalized

	return distances



def prune(graph, deviations=1.5) :
	""" Removes all edges with a distance higher than 'deviations' standard
	    deviations from the mean
	"""
	# Get weights (throws an error if they don't exist)
	weights = graph.edge_properties["weights"]

	# First find the edges that will be removed
	u = numpy.mean(weights.fa)
	sd = numpy.sqrt(numpy.var(weights.fa))
	above_treshold = graph.new_edge_property("bool")
	above_treshold.fa = weights.fa < u - deviations*sd

	# Then construct a graphView reflecting this
	filtered = gt.GraphView(graph, efilt=above_treshold)

	# Prune vertixes that have no outgoing edges
	connected = graph.new_vertex_property("bool")
	connected.fa = [v.in_degree() + v.out_degree() > 0 for v in filtered.vertices()]

	# Print out how many nodes we filter away
	print(str(sum(1 - connected.fa)) + " nodes pruned")


	# Return a graphview reflecting this
	return gt.GraphView(filtered, vfilt=connected)



def cluster(graph, nb_clusters=20, iter=100) :

	# Get weights (throws an error if they don't exist)
	weights_normalized = graph.edge_properties["weights_normalized"]

	return gt.community_structure(graph, iter, nb_clusters, weight=weights_normalized)



def getCluster(graph, clusters, cluster_index) :

	# Create a filter
	in_cluster = graph.new_vertex_property("bool")
	in_cluster.fa = [(clusters[v] == cluster_index) for v in graph.vertices()]

	# Create graph_view for the cluster
	return gt.GraphView(graph, vfilt=in_cluster)



def show(graph, clusters="orange", filename="graph.png") :

	# Get indices
	indices = graph.vertex_properties["indices"]

	# Define class colors
	class_colors = graph.new_vertex_property("vector<float>")
	colors = [[0.1, 0.1, 0.1, 0.9], [0.3, 0.3, 0.3, 0.9], [0.5, 0.5, 0.5, 0.9]] 
	for v in graph.vertices() : class_colors[v] = colors[indices[v]]

	# Get weights and positions
	weights_normalized = graph.edge_properties["weights_normalized"]
	pos = gt.sfdp_layout(graph, eweight=weights_normalized)

	# Print graph to file
	gt.graph_draw(graph, pos=pos, output_size=(1000, 1000), vertex_halo=True, vertex_halo_color=class_colors, vertex_color=clusters,
			   vertex_fill_color=clusters, vertex_size=5, edge_pen_width=weights_normalized, output=filename)



def showClusters(graph, clusters, filename="graph_clusters.png") :
	# Prune inter cluster edges
	intra_cluster = graph.new_edge_property("bool")
	intra_cluster.fa = [(clusters[e.source()] == clusters[e.target()]) for e in graph.edges()]

	# Create graph with less edges
	g_cluster = gt.GraphView(graph, efilt=intra_cluster)

	show(g_cluster, clusters, filename=filename)





def showOnImages(graph, images) :
	""" Displays the feature points of the graph as they are located on the images
	    Input: graph [Graph]
		       images [List of images]
	"""

	def tails(it):
		""" tails([1,2,3,4,5]) --> [[1,2,3,4,5], [2,3,4,5], [3,4,5], [4,5], [5], []] """
		while True:
			tail, it = tee(it)
			yield tail
			next(it)

	# Interpolate images to double size
	scale = 2.0

	# Image paths
	bg_path = "graph_background.png"
	fg_path = "graph_foreground.png"
	merge_path = "graph_on_image.png"

	# Put images together and resize
	bg_small = numpy.concatenate(images, axis=1)
	bg = imresize(bg_small, size=scale, interp='bicubic')
	pylab.imsave(bg_path, bg)

	# Calculate offsets
	offsets = map(sum, [list(t) for t in tails(map(lambda i : i.shape[1]*scale, images))])[::-1]

	# Get scaled positions
	ind_prop = graph.vertex_properties["indices"]
	positions = graph.vertex_properties["positions"]
	positions_scaled = graph.new_vertex_property("vector<float>")
	for v in graph.vertices() : positions_scaled[v] = numpy.array(positions[v]) * scale + numpy.array([offsets[ind_prop[v]], 0])

	# Draw graph
	gt.graph_draw(graph, 
				  pos=positions_scaled, 
				  fit_view=False, 
				  output_size=[bg.shape[1], bg.shape[0]],
				  vertex_size=8,
				  vertex_fill_color=ind_prop,
				  output=fg_path
				 )
	
	# Merge the graph and background images
	background = Image.open(bg_path)
	foreground = Image.open(fg_path)
	background.paste(foreground, (0, 0), foreground)
	background.save(merge_path)
	
	# Show resulting image
	im = pylab.imread(merge_path)
	pylab.imshow(im)






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


def partitionKeypoints(graph, keypoints, clusters) :
	""" Partitions the keypoints up in two sets (one for each image) for each
	    cluster.
		Input: graph [Graph] The graph containing indices of keypoints
		       keypoints [List of Keypoints]
		       clusters [PropertyMap of the Graph]
	"""
	indices = graph.vertex_properties["indices"]

	d1 = ((clusters[v], v, keypoints[graph.vertex_index[v]]) for v in graph.vertices())
	d2 = ((n, [(v,k) for (i,v,k) in g]) for n,g in groupby(sorted(d1), lambda e : e[0]))
	d3 = ((i, ((n, ((v,k) for (v,k) in g)) for n,g in groupby(ks, lambda e : indices[e[0]]))) for (i,ks) in d2)
	d4 = [[[i for i in l] for (j,l) in ll] for (i,ll) in d3]
	return d4
