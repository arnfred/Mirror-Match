"""
Python module for use with openCV and features.py to display keypoints
and their links between images

Most of this code is adapted from Jan Erik Solem's python wrapper 
(http://www.janeriksolem.net/2009/02/sift-python-implementation.html)

Jonas Toft Arnfred, 2013-03-08
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import cv2
import pylab
import numpy



####################################
#                                  #
#           Functions              #
#                                  #
####################################

def appendimages(im1, im2) :
	""" return a new image that appends the two images side-by-side.
	"""
	return numpy.concatenate((im1,im2), axis=1)



def getPositions(keypoints) : 
    pos = numpy.zeros([len(keypoints), 2])
    for i in range(len(keypoints)) :
        pos[i][0] = keypoints[i].pt[0]
        pos[i][1] = keypoints[i].pt[1]
    return pos



def keypoints(im, locs) :
	""" show image with features. input: im (image as array), 
	    locs (row, col, scale, orientation of each feature) 
	"""
	# Extract positions from keypoints
	pos = getPositions(locs)

	# Plot all keypoints
	pylab.gray()
	pylab.imshow(im)
	pylab.plot([p[1] for p in pos], [p[0] for p in pos], '.b')
	pylab.axis('off')
	pylab.show()



def matches(im1, im2, locs1, locs2, matchpos) :
	""" show a figure with lines joining the accepted matches in im1 and im2
	    input: im1,im2 (images as arrays), locs1,locs2 (location of features), 
	    matchscores (as output from 'match'). 
	"""
	# Extract positions from keypoints
	pos1 = getPositions(locs1)
	pos2 = getPositions(locs2)

	# Construct unified image
	im3 = appendimages(im1,im2)

	# Display image
	pylab.gray()
	pylab.imshow(im3)

	# Plot all lines
	cols1 = im1.shape[1]
	for i in range(len(matchpos)):
		if matchpos[i] != None:
			pylab.plot([pos1[i,1], pos2[int(matchpos[i]),1]+cols1], [pos1[i,0], pos2[int(matchpos[i]),0]], 'c')
	pylab.axis('off')
	pylab.show()
