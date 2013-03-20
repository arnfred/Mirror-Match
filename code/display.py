"""
Python module for use with openCV and features.py to display keypoints
and their links between images

Most of this code is adapted from Jan Erik Solem's python wrapper for sift
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
#             Images               #
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


def compareKeypoints(im1, im2, locs1, locs2) :
	""" Show two images next to each other with the keypoints marked
	"""

	# Extract positions from keypoints
	pos1 = getPositions(locs1)
	pos2 = getPositions(locs2)
	print(pos1)
	print(pos2)

	# Construct unified image
	im3 = appendimages(im1,im2)

	# Find the offset and add it
	offset = im1.shape[1]
	pos2[:,1] = pos2[:,1] + offset

	# Show images
	pylab.gray()
	pylab.imshow(im3)
	pylab.plot([p[1] for p in pos1], [p[0] for p in pos1], '.b')
	pylab.plot([p[1] for p in pos2], [p[0] for p in pos2], '.b')
	pylab.axis('off')
	pylab.show()



def matches(im1, im2, locs1, locs2, matchpos) :
	""" show a figure with lines joining the accepted matches in im1 and im2
	    input: im1,im2 (images as arrays), locs1,locs2 (location of features), 
	    matchscores (as output from 'match'). 
	"""

	assert len(locs1) == len(matchpos)

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



def matchesWithMask(images, keypoints, matchPos, mask) :
	""" Display only those matches that are masked
	    input: images [A list containing two images] The two images you want to plot
		       keypoints [A list containing two keypoints] The keypoints in use
		       matchPos [A list of indices] matchPos[0] is the keypoint in keypoints[1] 
		                                    that match with keypoints[0][0]
		       mask [array of booleans] If mask[i] == true then keypoints[0][i] is displayed
	"""
	# Take the n highest scoring points and plot them
	masked_matchPos = [m if b else None for (m,b) in zip(matchPos, mask)]

	# Show result
	matches(images[0], images[1], keypoints[0], keypoints[1], masked_matchPos)



####################################
#                                  #
#            Plotting              #
#                                  #
####################################


def barWithMask(X,Y,mask,color='blue') :
	""" Show a barplot, but make the bars corresponding to the mask stand out
	    input: X [array of numbers] (x-values of the data)
		       Y [array of numbers] (y-values of the data)
			   mask [array of booleans] (if mask[i] == true, then Y[i] is emphatized)
			   color [string] (the preferred color)
	"""
	# Show scores in subplot
	margin = (max(Y) - min(Y)) * 0.15
	x_min, x_max = min(X),max(X)
	y_min, y_max = min(Y)-margin, max(Y)+margin

	alpha = [0.5, 0.7]
	alphaMap = [alpha[1] if b else alpha[0] for b in mask]
	for x, y, a in zip(X,Y,alphaMap) :
		pylab.bar(x, y, color=color, alpha=a)
		
	pylab.xlim(x_min, x_max)
	pylab.ylim(y_min, y_max)

	# Remove uneeded axices
	ax = pylab.gca()
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')


def scoreHist(scores) : 

	same = [score for (b, score) in scores if b]
	diff = [score for (b, score) in scores if not b] 

	mean_same = numpy.mean(same)
	mean_diff = numpy.mean(diff)
	variance = numpy.mean([numpy.var(same), numpy.var(diff)])
	sd = numpy.sqrt(variance)
	print("Mean(same):\t\t{0:.3f}".format(mean_same))
	print("Mean(diff):\t\t{0:.3f}".format(mean_diff))
	print("Diff of means:\t\t{0:.3f}".format(numpy.abs(mean_diff - mean_same)))
	print("Standard deviation:\t{0:.3f}".format(sd)) 
	print("# of sd's:\t\t{0:.3f}".format(numpy.abs(mean_diff - mean_same)/sd))

	x_min = min(min([same,diff])) 
	x_max = max(max([same, diff]))
	margin = (x_max - x_min) * 0.2

	pylab.subplot(1,2,1)
	pylab.hist(same, bins=20, label="Same", color="green", alpha=0.65)
	pylab.legend()
	pylab.xlim(x_min - margin ,x_max + margin )
	removeDecoration

	pylab.subplot(1,2,2)
	pylab.hist(diff, bins=20,label="Diff", color="red", alpha=0.65)
	pylab.legend()
	pylab.xlim(x_min - margin ,x_max + margin )
	removeDecoration



####################################
#                                  #
#              Util                #
#                                  #
####################################

def makeMask(f,ls) : return [f(k) for k in ls]

def removeDecoration() :
	# Remove uneeded axices
	ax = pylab.gca()
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')

