"""
Python module for creating a test set from source image pairs

Jonas Toft Arnfred, 2013-06-05

The code is released under the following license:

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

import random
from string import ascii_uppercase, digits
from os.path import isdir, dirname, exists
from os import makedirs
from scipy.misc import imresize
import pylab
import numpy
import sys
import getopt
import cv2


####################################
#                                  #
#           Functions              #
#                                  #
####################################

def loadImage(path) : 
	""" Given a path, an image will be loaded and converted to grayscale
		input: path [string] (path to the image)
		out:   [numpy.ndarray] cv2 representation of image in one channel
	"""
	# Try to read image, and if doesn't exist, throw exception
	img = cv2.imread(path)
	if (img == None) : raise NonExistantPath(path, "Image doesn't exist")

	# Convert to grayscale: First we convert the image to the L*u*v color space
	# and then return the luminance channel
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:,:,0]

	# Preprocess image
	#img_processed = preprocessing.faceNorm(img_gray)

	return img_gray


def getRandPath(folder, size=6, chars=ascii_uppercase + digits) :
	condition = True
	while condition:
		name = ''.join(random.choice(chars) for x in range(size))
		condition = isdir(folder + name)
	return folder + name


def makeTestPair(paths, homography, collection, width=250, height=250, scale = 1.0) :
	
	images = map(loadImage, paths)
	
	# Crop part of first image and part of second image:
	(top_o, left_o) = (random.randint(0, images[0].shape[0]-height), random.randint(0, images[0].shape[1]-width))
	(top_n, left_n) = (random.randint(0, images[1].shape[0]-height), random.randint(0, images[1].shape[1]-width))
	
	# Get two file names
	c_path = getRandPath("%s/" % collection)
	print(c_path)
	if not exists(dirname(c_path)) : makedirs(dirname(c_path))
		
	# Make sure we save as gray
	pylab.gray()
	
	im1 = images[0][top_o: top_o + height, left_o: left_o + width]
	im2 = images[1][top_n: top_n + height, left_n: left_n + width]
	im1_scaled = imresize(im1, size=float(scale), interp='bicubic')
	im2_scaled = imresize(im2, size=float(scale), interp='bicubic')
	pylab.imsave(c_path + "_1.jpg", im1_scaled)
	pylab.imsave(c_path + "_2.jpg", im2_scaled)
	#imsave(c_path + "_1.jpg", im1)
	#imsave(c_path + "_2.jpg", im2)
	
	T1 = numpy.identity(3)
	T1[0,2] = left_o
	T1[1,2] = top_o
	
	T2 = numpy.identity(3)
	T2[0,2] = -1*left_n# * scale
	T2[1,2] = -1*top_n# * scale
	
	Ts = numpy.identity(3)
	Ts[0,0] = scale
	Ts[1,1] = scale
	
	Tsinv = numpy.identity(3)
	Tsinv[0,0] = 1.0/scale
	Tsinv[1,1] = 1.0/scale
	
	hom = Ts.dot(T2).dot(homography).dot(T1).dot(Tsinv)
	hom = hom / hom[2,2]
	numpy.savetxt(c_path, hom)
	
	return c_path


def getSavedHomography(paths) :
	def getHomographyPath(paths) :
		d = "".join(paths[0].split('/')[0:-1])
		if d == "" : d = "."
		p1 = paths[0].split('/')[-1].split(".")[0]
		p2 = paths[1].split('/')[-1].split(".")[0]
		return "%s/%s_%s_hom.npy" % (d, p1, p2)
	
	try :
		homography_path = getHomographyPath(paths)
		print(homography_path)
		with open(homography_path) :
			homography = numpy.loadtxt(homography_path)
	# In case the file doesn't exist, try to estimate something
	except IOError :
		raise Usage("no homography for paths: %s, %s" % (paths[0], paths[1]))
	
	return homography



####################################
#                                  #
#              Main                #
#                                  #
####################################

# from http://www.artima.com/weblogs/viewpost.jsp?thread=4829
class Usage(Exception):
	def __init__(self, msg):
		self.msg = msg

def main(argv=None):
	""" Create a testset consisting of n image pairs (default 100) based on the source pair.
	    Usage example: python create_testsets -n 150 'fairey_lady' 'New Testset'
		This will create 100 image pairs and homographies in the directory 'New Testset'. 
		If 'New Testset' already exists, 100 image pairs and homographies will be added.
	"""
	if argv is None:
		argv = sys.argv
	try:
		try:
			opts, args = getopt.getopt(argv[1:], "hn:", ["help"])
		except getopt.error, msg:
			 raise Usage(msg)

		if len(args) == 3:
			p1 = args[0]
			p2 = args[1]
			d = args[2]
		elif len(args) == 2:
			p1 = "%s_1.jpg" % args[0]
			p2 = "%s_2.jpg" % args[0]
			d = args[1]
		else : raise Usage("Too many or two few arguments. Use --help for instructions")

		# Get number of patches
		n = 100
		for (o,v) in opts : 
			if o == '-n' : 
				try :
					n = int(v)
				except ValueError :
					raise Usage("Value for -n not an integer: %s" % v)
			if o == '-h' :

				t = """Create a testset consisting of n image pairs (default 100) based on the
source pair. Usage example: 

python create_testsets -n 150 'fairey_lady' 'New Testset'

This will create 100 image pairs and homographies in the directory 'New
Testset'. If 'New Testset' already exists, 100 image pairs and homographies
will be added.""" 
				print(t) 
				return 0


		# Check that paths exists
		if (not exists(p1) or not exists(p2)) : raise Usage("Unexistant paths: %s, %s" % (p1,p2))

		homography = getSavedHomography([p1,p2])
		test_path = [makeTestPair([p1,p2], homography, d, scale = 1.0) for _ in range(n)]

	except Usage, err:
		print >>sys.stderr, err.msg
		print >>sys.stderr, "for help use --help"
		return 2


# Excute program
if __name__ == "__main__":
    sys.exit(main())
