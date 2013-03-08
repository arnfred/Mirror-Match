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



def features(im, locs) :
	""" show image with features. input: im (image as array), 
	    locs (row, col, scale, orientation of each feature) 
	"""
	pylab.gray()
	pylab.imshow(im)
	pylab.plot([p[1] for p in locs], [p[0] for p in locs], '.b')
	pylab.axis('off')
	pylab.show()



def matches(im1, im2, locs1, locs2, matchscores) :
	""" show a figure with lines joining the accepted matches in im1 and im2
		input: im1,im2 (images as arrays), locs1,locs2 (location of features), 
		matchscores (as output from 'match'). """

	im3 = appendimages(im1,im2)

	pylab.gray()
	pylab.imshow(im3)

	cols1 = im1.shape[1]
	for i in range(len(matchscores)):
		if matchscores[i] != None:
			pylab.plot([locs1[i,1], locs2[int(matchscores[i]),1]+cols1], [locs1[i,0], locs2[int(matchscores[i]),0]], 'c')
	pylab.axis('off')
	pylab.show()
