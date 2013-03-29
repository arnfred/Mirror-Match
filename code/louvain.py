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
from itertools import groupby

####################################
#                                  #
#           Functions              #
#                                  #
####################################



def cluster(graph, verbose=False) :
	moved = 1
	partitions = initL(graph)
	w = graph.ep["weights"]
	K = 2*numpy.sum(w.fa)
	while (moved > 0) :
		moved = sum([move(graph, v, partitions, K) for v in graph.vertices()])
		if verbose : print("Moved %i vertices" % moved)
	p = reassign(partitions)
	t = [(k, len(list(g))) for k,g in groupby(sorted(p.fa))]
	return p,len(t)


def initL(graph) :
	partition = graph.new_vertex_property("int")
	partition.fa = numpy.arange(0,graph.num_vertices())
	return partition



def deltaQ(graph, v, partition, partitions, K) :
	def toPartition(e) : 
		return (partitions[e.target()] == partition) or (partitions[e.source()] == partition)
	w = graph.ep["weights"]
	k = graph.vp["degrees"]
	v_weight = 2 * numpy.sum([w[e] for e in v.all_edges() if toPartition(e)])
	k_neighbours = numpy.sum((partitions.fa == partition) * k.fa)
	return 1.0/K * (v_weight - (k[v] * k_neighbours) / K)


def neighbour_partitions(graph, v, partitions) :
	return set([partitions[n] for n in v.all_neighbours()])



def prospect_set(graph, v, orig_partition, partitions, K) :
	return [(deltaQ(graph, v, p, partitions, K),p) for p in neighbour_partitions(graph, v, partitions) if p != orig_partition]



def move(graph, v, partitions, K) :
	orig_partition = partitions[v]
	# move v out of orig_partition
	partitions[v] = -1
	# See how much we would gain from moving p back to orig_partition
	old_prospect = deltaQ(graph, v, orig_partition, partitions, K)
	# What is the best gain elsewhere?
	new_prospects = prospect_set(graph, v, orig_partition, partitions, K)
	max_prospect = max(new_prospects) if (len(new_prospects) > 0) else (-1,-1)
	# Is it worth moving?
	delta = max_prospect[0] - old_prospect
	if (delta > 0) : 
		partitions[v] = max_prospect[1]
		return 1
	else : 
		partitions[v] = orig_partition
		return 0





def reassign(partitioning) :
	assignments = [(i,k) for i,(k,g) in enumerate(groupby(sorted(partitioning.fa)))]
	p_new = partitioning.copy()
	for i,k in assignments :
		p_new.fa[partitioning.fa == k] = i
	return p_new
