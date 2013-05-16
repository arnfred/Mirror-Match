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
		moved = sum([1 for d in deltas if d > 0])
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
		Input: Weights [numpy.darray] Adjecency matrix of the graph
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
