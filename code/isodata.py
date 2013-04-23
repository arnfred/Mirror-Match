"""
Python module implementing the isodata algorithm as explained in
 http://www.cs.umd.edu/~mount/Projects/ISODATA/
 http://www.cs.umd.edu/~mount/Projects/ISODATA/ijcga07-isodata.pdf (page 5-6)

Original publication:

	Ball, Geoffrey H., Hall, David J. (1965) Isodata: a method of data analysis and
	pattern classification, Stanford Research Institute, Menlo Park,United States.
	Office of Naval Research. Information Sciences Branch

	http://www.dtic.mil/cgi-bin/GetTRDoc?Location=U2&doc=GetTRDoc.pdf&AD=AD0699616

Jonas Toft Arnfred, 2013-04-23

The code is released under the following license. I'm not the inventor of this 
algorithm, so I'm just putting the license in here so others can benefit from the
source code. Here goes:

	Copyright (C) 2013 Jonas Toft Arnfred

	Permission is hereby granted, free of charge, to any person obtaining a copy of
	this software and associated documentation files (the "Software"), to deal in
	the Software without restriction, including without limitation the rights to
	use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
	of the Software, and to permit persons to whom the Software is furnished to do
	so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

from itertools import combinations
import random
import numpy



####################################
#                                  #
#           Functions              #
#                                  #
####################################


# If you are looking at the paper, these are my more human readable versions of
# the parameters:
# n_min = min_partition_size
# I_max = max_iterations
# rho_max = max_sd

def cluster(points, k_init=None, max_iterations=100, min_partition_size=5, max_sd=10, min_distance=20, max_mergers=10, seed=1864) :
	""" Cluster a set of points according to the isodata algorithm.
	    The inputs are various and not normalized
	    Input: points [numpy.ndarray] Each row is a datapoint
	           k_init [int] The initial amount of clusters
	           max_iterations [int] The initial amount of clusters
	           min_partition_size [int] How many data points a cluster must have
	           max_sd [int] The maximum standard variation we accept within a cluster before we split it
	           min_distance [int] The minimum distance between two clusters before we consider a merge
	           max_mergers [int] Maximum amount of mergers per round
	           seed [int] Specify a random seed
	"""
	
	# Set a sensible k_init if it wasn't specified
	if k_init == None:
		k_init = points.shape[0] / (min_partition_size * 2)
	
	# Sample nb_partitions points as initial guess
	random.seed(seed)
	centers = numpy.array(random.sample(points, k_init))
	
	# Iterate
	for i in range(max_iterations) :
		
		# As long as we remove clusters, we should reassign points
		removed = 1
		while (removed > 0) :
			
			#[Step 2] Get partitioning according to centers
			partitioning = pickCenter(centers, points)
			k = len(set(partitioning))
			
			#[Step 3] Remove partitions that are too small
			removed = removePartitions(partitioning, min_partition_size)
			
			#[Step 4] Recalculate center
			centers = getCenters(partitioning, points)
		
		
		# [Step 5] Find average distances
		spread = getPartitionSpread(partitioning, centers, points)
		
		# [Step 6] Test if we should do step 8 and 9:
		# I quote: "If this is the last iteration, then set Lmin = 0 
		#           and go to Step 9. Also, if 2k > k_init and it is 
		#           either an even numbered iteration or k >= 2*k_init , 
		#           then go to Step 9"
		test_1 = i == (max_iterations-1)
		test_2 = 2 * k > k_init and (i % 2 == 0)
		test_3 = k >= 2 * k_init
		
		if not (test_1 or test_2 or test_3) :
			
			# [Step 7] Measure the standard deviation of each dimension for each cluster
			sds = getSD(partitioning, centers, points)
			
			# Count how many splits we perform
			split = 0
			
			# [Step 8] For each partition check if it should be split
			for j, p in enumerate(set(partitioning)) :
				
				if splitTest(j, sds[j], max_sd, spread, numpy.sum(partitioning == p), min_partition_size, k, k_init) :
					# Split the partition according to the highest sd direction
					split += 1
					centers = splitPartition(centers, sds[j], j, k)

			if split != 0 : continue
		
		# Only merge if this isn't the last iteration (see [Step 6])
		if i != (max_iterations-1) :
			
			# [Step 9] Get distances between all clusters
			ds_pairs = partitionDist(centers)
			
			# [Step 10] Merge pairs that are below a certain treshold
			mergers = mergePartitions(ds_pairs, centers, partitioning, min_distance, max_mergers)
	
	# [Step 11] Eventually when all iterations have been completed, return the partitions
	return partitioning



####################################
#                                  #
#       Private Functions          #
#                                  #
####################################


def mergePartitions(ds_pairs, centers, partitioning, min_distance, max_mergers) :
	
	# Check if i and j are already in the set and add them to the set
	def hasMerged(s, i, j) :
		has_already = i in s or j in s
		s.add(i)
		s.add(j)
		return has_already
	
	def mergeCenters(i,j) :
		n_i = numpy.sum(partitioning == i)
		n_j = numpy.sum(partitioning == j)
		centers_old = centers[(numpy.arange(k) != j) & (numpy.arange(k) != i)]
		centers_new = numpy.array([(1.0 / (n_i + n_j)) * (n_i * centers[i] + n_j * centers[j])])
		return numpy.concatenate((centers_old, centers_new))
	
	# Find m (= max_mergers) partitions that are within min_distance of each other
	merge_init = [d for i,d in enumerate(ds_pairs) if d[0] < min_distance and i < max_mergers]
	
	# Make sure we don't merge any group twice
	merge_set = set()
	merge_pairs = [(d,(i,j)) for (d,(i,j)) in merge_init if hasMerged(merge_set, i, j)]
	
	# Now merge the remaining partitions
	mergers = 0
	for (d, (i,j)) in merge_pairs : 
		mergeCenters(i,j)
		mergers += 1
	
	return mergers



def partitionDist(centers) :
	k = centers.shape[0]
	def d(i,j) : return numpy.linalg.norm(centers[i] - centers[j])
	return sorted([(d(i,j), (i,j)) for i,j in combinations(range(k),2)])



def splitPartition(centers, sd, j, k) :
	# Find the delta vector to add and subtract from the old center
	delta = numpy.zeros(sd.shape)
	delta[numpy.argmax(sd)] = numpy.max(sd)
	# Remove the old center and add the two new ones
	centers_old = centers[numpy.arange(k) != j]
	centers_new = numpy.array([centers[j] + delta, centers[j] - delta])
	return numpy.concatenate((centers_old, centers_new))



def splitTest(j, sd_j, max_sd, spread, n_j, min_partition_size, k, k_init) :
	avg_spread = numpy.mean(spread)
	test_1 = numpy.max(sd_j) > max_sd
	test_2_1 = spread[j] > avg_spread
	test_2_2 = n_j > 2*(min_partition_size + 1)
	test_2 = test_2_1 and test_2_2
	test_3 = k <= (k_init / 2.0)
	return test_1 and (test_2 or test_3)



def getSD(partitioning, centers, points) :
	def sd(points, center) : return numpy.sqrt(numpy.var(points - center, axis = 0))
	sds = [sd(points[partitioning == p], centers[p]) for p in set(partitioning)]
	return numpy.array(sds)



def getPartitionSpread(partitioning, centers, points) :
	def getDists(rows) : return [numpy.linalg.norm(row) for row in rows]
	spread = [numpy.mean(getDists(points[partitioning == p] - centers[p])) for p in set(partitioning)]
	return numpy.array(spread)



def getCenters(partitioning, points) :
	def getCenter(p) : return numpy.mean(points[partitioning == p], axis=0)
	return numpy.array([getCenter(p) for p in set(partitioning) if p != -1])



def removePartitions(partitioning, treshold) :
	# Counter for how many partitions where removed
	removed = 0
	
	# Check for partitions that are too small
	for p in range(numpy.max(partitioning)) :
		if numpy.sum(partitioning == p) < treshold :
			partitioning[partitioning == p] = -1
			removed += 1
	
	return removed



def pickCenter(centers, points) :
	
	def closestCenter(centers, p) :
		ds = [numpy.linalg.norm(c - p) for c in centers]
		return numpy.argmin(ds)
	
	return numpy.array([closestCenter(centers, p) for p in points])
