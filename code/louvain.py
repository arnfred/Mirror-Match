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
    delta_Q = 1
    m_diff = 1
    #while (moved > 0) :
    while (m_diff > 0 and moved > 0) :
        deltas = [move(weights, i, partitions, K, False) for i in indices]
        moved = sum([1 for d in deltas if d > 0])
        delta_Q = sum(deltas)
        m_old = m
        m = modularity(weights, partitions)
        m_diff = m - m_old
        if verbose : 
            print("Moved %i vertices (modularity now: %.6f, change of %.6f (%.6f))" % (moved,m,m_diff,delta_Q))
    return partitions


def move(weights, index, partitions, K, verbose = False) :
    # Move out of partition
    old_partition = partitions[index]
    #partitions[index] = -1
    # See how much we would gain from moving p back to orig_partition
    lost_Q = deltaQ(weights, index, (old_partition == partitions), K)
    # What is the best gain elsewhere?
    prospects = set(partitions[numpy.nonzero(weights[index])])
    prospect_Q = [(deltaQ(weights, index, (p == partitions), K), p) for p in prospects]
    max_gained_Q = max(prospect_Q) if (len(prospect_Q) > 0) else (-1,-1)
    # Is it worth moving?
    delta = max_gained_Q[0] - lost_Q
    if (delta > 0.000001) : 
        if verbose :
            print("Delta is %.8f, moving node from partition %i to partition %i" % (delta, old_partition, max_gained_Q[1]))
            print(weights[partitions == old_partition][:,partitions == old_partition])
            print(weights[partitions == max_gained_Q[1]][:,partitions == max_gained_Q[1]])
        partitions[index] = max_gained_Q[1]
        return delta
    else : 
        partitions[index] = old_partition
        return 0


# Gain in Q from moving index to new_partition
def deltaQ(weights, index, partition, K) :
    v_edges = weights[index]
    v_weight = 2.0 * v_edges[partition].sum()
    p_weight = weights[partition]
    k_neighbours = 2.0 * p_weight.sum() - p_weight[:, partition].sum()
    k_v = v_edges.sum()
    Q_iv = 1.0/K * (v_weight - (k_v * k_neighbours + k_v**2) / K)
    #print("new: %.6f, old: %.6f" % (Q_iv, Q_iv_old))
    return Q_iv


def modularity(weights, partitions) :
    """ Calculates the modularity of the partition partitioned by 'partition' in the weight matrix
        Input: Weights [numpy.darray] Adjecency matrix of the graph
               partition [boolean numpy.darray] A partition marking the partition of vertices 
    """
    K = weights.sum()
    indices = numpy.arange(0,weights.shape[0])
    def mod_part(partition) :
        internal_sum = weights[partition][:, partition].sum()
        external_sum = weights[partition].sum()
        fraction = (internal_sum / (K))
        E_fraction = (external_sum / (K)) ** 2
        return fraction - E_fraction

    ms = [mod_part(partitions==p) for p in set(partitions)]
    return sum(ms)
