"""
Python module for preprocessing images to make them lighting invariant.
The module is for use with experiments using descriptors such as BRIEF
and ORB.

The code is adapted from different sources. This is marked in the
descriptions of the individual functions.

Jonas Toft Arnfred, 2013-03-07
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import numpy
from scipy import ndimage
from pylab import imshow


####################################
#                                  #
#           Functions              #
#                                  #
####################################

def stretch(im) :
	""" Stretches the histogram of an image.
	    Input: im [numpy.ndarray] Image in uint or float format
		Output: [numpy.ndarray] Float version of the input image
	                            normalized between 0 and 1
	"""
	im_float = numpy.array(im - numpy.min(im), numpy.float32)
	return im_float / numpy.max(im_float)



def toUInt(im) :
	""" Converts a floating point image to integer image between 0 and 255
	    Input: im [numpy.ndarray] Image in uint or float format
	    Output: [numpy.ndarray] integer version of the image
	"""
	stretched = stretch(im)
	return numpy.array(numpy.floor(stretched*255), numpy.uint8)



def faceNorm(im, contrast = 0.2) :
	""" Calculates a lighting neutral image
	    Input: im [numpy.ndarray] Image in whatever format
		Output: [numpy.ndarray] UINT version of image
	"""
	alpha = 0.1
	tau = 10.0
	gamma = 0.2
	sigma1 = 1.0
	sigma2 = 3.0

	im = stretch(im)
	c = ndimage.gaussian_filter(im, sigma1)
	s = ndimage.gaussian_filter(im, sigma2)
	q = numpy.asarray(c - s)
	w = numpy.asarray(c+s+0.000001)
	nDoG = q/w

	A = contrast # The smaller the greater the enhancement
	B = 1
	w = nDoG*(A+B)
	ww = numpy.abs(nDoG)+A
	cenDoG = w/ww

	return toUInt(cenDoG)



def invert(im) :
	""" Inverts an image so light turns dark & vice versa
	    Input: im [numpy.ndarray] Input image in normalized floating point
	    Output: [numpy.ndarray] image in normalized floating point
	"""
	return (im-1)*-1.0



def tanTriggs(im, contrast = 0.1) :
	""" Calculates a lighting neutral image
	    Input: im [numpy.ndarray] Image in whatever format
		Output: [numpy.ndarray] Floating point version of image
	"""
	alpha = contrast
	tau = 10.0
	gamma = 0.2
	sigma0 = 1.0
	sigma1 = 2.0

	im = invert(stretch(im))
	im = numpy.power(im, gamma)
	im = numpy.asarray(ndimage.gaussian_filter(im, sigma1) - ndimage.gaussian_filter(im,sigma0))
	im = im / numpy.power(numpy.mean(numpy.power(numpy.abs(im), alpha)), 1.0 / alpha)
	im = im / numpy.power(numpy.mean(numpy.power(numpy.minimum(numpy.abs(im), tau), alpha)), 1.0 / alpha)
	im = tau * numpy.tanh(im / tau)
	return toUInt(im)
