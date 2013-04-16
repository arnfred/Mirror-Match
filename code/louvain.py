"""
Python module for use with Graph-Tool to cluster a graph based on the louvain 
clustering algorithm. Read more here: 

http://iopscience.iop.org/1742-5468/2008/10/P10008 and
http://arxiv.org/pdf/0803.0476

Jonas Toft Arnfred, 2013-03-28
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import numpy
#from itertools import groupby

####################################
#                                  #
#           Functions              #
#                                  #
####################################


def cluster(weights, verbose=False) :
	indices = numpy.arange(0, weights.shape[0])
	partitions = indices.copy()
	K = numpy.sum(weights)
	moved = 1
	m = 0
	while (moved > 0) :
		deltas = [move(weights, i, partitions, K) for i in indices]
		moved = sum([1 if d > 0 else 0 for d in deltas])
		if verbose : 
			m_new = modularity(weights, partitions)
			m_diff = m_new - m
			m = m_new
			print("Moved %i vertices (modularity now: %.4f, change of %.4f)" % (moved,m,m_diff))
	return partitions


def move(weights, index, partitions, K) :
	# Move out of partition
	old_partition = partitions[index]
	partitions[index] = -1
	# See how much we would gain from moving p back to orig_partition
	lost_Q = deltaQ(weights, index, (old_partition == partitions), K)
	# What is the best gain elsewhere?
	prospects = set(partitions[numpy.nonzero(weights[index])])
	prospect_Q = [(deltaQ(weights, index, (p == partitions), K), p) for p in prospects]
	max_gained_Q = max(prospect_Q) if (len(prospect_Q) > 0) else (-1,-1)
	# Is it worth moving?
	delta = max_gained_Q[0] - lost_Q
	if (delta > 0) : 
		partitions[index] = max_gained_Q[1]
		return delta
	else : 
		partitions[index] = old_partition
		return 0

# Gain in Q from moving index to new_partition
def deltaQ(weights, index, partition_mask, K) :
	v_edges = weights[index]
	v_weight = 2.0 * numpy.sum(v_edges[partition_mask])
	k_neighbours = numpy.sum(weights[partition_mask])
	k_v = numpy.sum(v_edges)
	return 1.0/K * (v_weight - (k_v * k_neighbours) / K)



def modularity(weights, partitions) :
	""" Calculates the modularity of the partition masked by 'mask' in the weight matrix
		Input: Weights [numpy.darray] The weightmatrix used
			   mask [boolean numpy.darray] A mask marking the partition of vertices 
	"""
	K = weights.sum()
	indices = numpy.arange(0,weights.shape[0])
	def mod_part(mask) :
		internal_sum = weights[mask][:, mask].sum()
		external_sum = weights[mask].sum()
		fraction = (internal_sum / (K))
		E_fraction = (external_sum / (K)) ** 2
		return fraction - E_fraction

	ms = [mod_part(partitions==p) for p in set(partitions)]
	return sum(ms)



# def cluster(graph, verbose=False) :
# 	moved = 1
# 	partitions = initL(graph)
# 	w = graph.ep["weights"]
# 	K = 2*numpy.sum(w.fa)
# 	while (moved > 0) :
# 		moved = sum([move(graph, v, partitions, K) for v in graph.vertices()])
# 		if verbose : print("Moved %i vertices" % moved)
# 	p = reassign(partitions)
# 	print(partitions)
# 	t = [(k, len(list(g))) for k,g in groupby(sorted(p.fa))]
# 	return p,len(t)
# 
# 
# def initL(graph) :
# 	partition = graph.new_vertex_property("int")
# 	partition.fa = numpy.arange(0,graph.num_vertices())
# 	return partition
# 
# 
# 
# def deltaQ(graph, v, partition, partitions, K) :
# 	def toPartition(e) : 
# 		return (partitions[e.target()] == partition) or (partitions[e.source()] == partition)
# 	w = graph.ep["weights"]
# 	k = graph.vp["degrees"]
# 	v_edges = [w[e] for e in v.all_edges()]
# 	v_edges_partition = [w[e] for e in v.all_edges() if toPartition(e)]
# 	v_weight = 2 * numpy.sum([w[e] for e in v.all_edges() if toPartition(e)])
# 	k_neighbours = numpy.sum((partitions.fa == partition) * k.fa)
# 	return 1.0/K * (v_weight - (k[v] * k_neighbours) / K)
# 
# 
# def neighbour_partitions(graph, v, partitions) :
# 	return set([partitions[n] for n in v.all_neighbours()])
# 
# 
# 
# def prospect_set(graph, v, orig_partition, partitions, K) :
# 	return [(deltaQ(graph, v, p, partitions, K),p) for p in neighbour_partitions(graph, v, partitions) if p != orig_partition]
# 
# 
# 
# def move(graph, v, partitions, K) :
# 	orig_partition = partitions[v]
# 	# move v out of orig_partition
# 	partitions[v] = -1
# 	# See how much we would gain from moving p back to orig_partition
# 	old_prospect = deltaQ(graph, v, orig_partition, partitions, K)
# 	# What is the best gain elsewhere?
# 	new_prospects = prospect_set(graph, v, orig_partition, partitions, K)
# 	max_prospect = max(new_prospects) if (len(new_prospects) > 0) else (-1,-1)
# 	# Is it worth moving?
# 	delta = max_prospect[0] - old_prospect
# 	if (delta > 0) : 
# 		partitions[v] = max_prospect[1]
# 		return 1
# 	else : 
# 		partitions[v] = orig_partition
# 		return 0
# 
# 
# 
# def reassign(partitioning) :
#     assignments = [(i,k) for i,(k,g) in enumerate(groupby(sorted(partitioning.fa)))]
#     p_new = partitioning.copy()
#     for i,k in assignments :
#         p_new.fa[partitioning.fa == k] = i
#     return p_new
