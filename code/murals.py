""" 

Python module for creating cropped pairs of images.  This code should be
accompanying the murals testset as found on
http://vintage.winklerbros.net/murals.html

Murals consist of 8 image pairs and homographies which can be found in 
the folder 'data'. The homographies are a 3 by 3 matrix and can be 
loaded using numpy.loadtxt. To use murals.py to create cropped pairs, 
you need to have python 2.8 and opencv on your system. Navigate to the 
directory where the files reside in a terminal and type:
    python murals.py IMAGE_PAIR TESTSET_NAME

where IMAGE_PAIR is the name of the image pair used to create crops. The 
following image pairs are available:
    banksy_city
    banksy_stroller
    blu_head
    blu_pencil
    fairey_burma
    fairey_lady
    houston
    scharf

Example:
    python murals.py fairey_lady My_Testset

The script supports the following options:
    --help, -h          show this help message and exit
    --location, -l      output directory (default: ".")
    --width, -x         crop width (default 250px)
    --height, -y        crop height (default 250px)
    --scale, -s         crop scale (default 1.0)
    --testset_size, -n  number of pairs in testset (default 100)

Example: Create 67 pairs and put them in the parent directory
    python murals.py -n 67 --location ".." fairey_lady My_Testset

This code as well as the images in the murals dataset are released under a
creative commons license: http://creativecommons.org/licenses/by-nc-nd/3.0/

If you use this work in your research please cite:

J. Arnfred, S. Winkler, S. S\"usstrunk
Mirror Match: Reliable Feature Point Matching Without Geometric Constraints.
Proc. 2nd Asian Conference on Pattern Recognition (ACPR), Naha, Japan, November 5-8, 2013.

As found here: http://vintage.winklerbros.net/Publications/acpr2013mm.pdf

Jonas Toft Arnfred, 2013-09-11
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import random
import numpy
import cv2
import sys, os, traceback, optparse
import time
import re
import pylab

from string import ascii_uppercase, digits
from os.path import isdir, dirname, exists
from os import makedirs
from scipy.misc import imresize

####################################
#                                  #
#           Functions              #
#                                  #
####################################

def getRandPath(folder, size=6, chars=ascii_uppercase + digits) :
    """ Construct a random path consisting of the chars provided """
    condition = True
    while condition:
        name = ''.join(random.choice(chars) for x in range(size))
        condition = isdir(folder + name)
    return folder + name


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

    return img_gray


def makeTestPair(paths, homography, collection, location=".", size=(250,250), scale = 1.0) :
    """ Given a pair of paths to two images and a homography between them,
        this function creates two crops and calculates a new homography.
        input: paths [strings] (paths to images)
               homography [numpy.ndarray] (3 by 3 array homography)
               collection [string] (The name of the testset)
               location [string] (The location (path) of the testset
               size [(int, int)] (The size of an image crop in pixels)
               scale [double] (The scale by which we resize the crops after they've been cropped)
        out:   nothing
    """
    
    # Get width and height
    width, height = size
    
    # Load images in black/white
    images = map(loadImage, paths)
    
    # Crop part of first image and part of second image:
    (top_o, left_o) = (random.randint(0, images[0].shape[0]-height), random.randint(0, images[0].shape[1]-width))
    (top_n, left_n) = (random.randint(0, images[1].shape[0]-height), random.randint(0, images[1].shape[1]-width))
    
    # Get two file names
    c_path = getRandPath("%s/%s/" % (location, collection))
    if not exists(dirname(c_path)) : makedirs(dirname(c_path))
        
    # Make sure we save as gray
    pylab.gray()
    
    im1 = images[0][top_o: top_o + height, left_o: left_o + width]
    im2 = images[1][top_n: top_n + height, left_n: left_n + width]
    im1_scaled = imresize(im1, size=float(scale), interp='bicubic')
    im2_scaled = imresize(im2, size=float(scale), interp='bicubic')
    pylab.imsave(c_path + "_1.jpg", im1_scaled)
    pylab.imsave(c_path + "_2.jpg", im2_scaled)
    
    # Homography for transpose
    T1 = numpy.identity(3)
    T1[0,2] = left_o
    T1[1,2] = top_o
    
    # Homography for transpose back
    T2 = numpy.identity(3)
    T2[0,2] = -1*left_n
    T2[1,2] = -1*top_n
    
    # Homography for scale
    Ts = numpy.identity(3)
    Ts[0,0] = scale
    Ts[1,1] = scale
    
    # Homography for scale back
    Tsinv = numpy.identity(3)
    Tsinv[0,0] = 1.0/scale
    Tsinv[1,1] = 1.0/scale
    
    # Combine homographyies and save
    hom = Ts.dot(T2).dot(homography).dot(T1).dot(Tsinv)
    hom = hom / hom[2,2]
    numpy.savetxt(c_path, hom)


def getSavedHomography(paths) :
    """ Utility function for loading a homography from file """

    def getHomographyPath(paths) :
        d = "/".join(paths[0].split('/')[0:-1])
        p1 = paths[0].split('/')[-1].split(".")[0]
        p2 = paths[1].split('/')[-1].split(".")[0]
        return "%s/%s_%s_hom.npy" % (d, p1, p2)
    

    homography_path = getHomographyPath(paths)
    print(homography_path)
    with open(homography_path) :
        homography = numpy.loadtxt(homography_path)

    return homography


def makeTestset(paths, homography, collection, n = 100, location=".", size=(250,250), scale = 1.0) :
    """ Create n test pairs in a given testset """
    [makeTestPair(paths, homography, collection, location = location, size = size, scale=scale) for _ in range(n)]



def main ():

    global options, args

    # Register path and collection from args
    path = args[0]
    paths = ("data/%s_1.jpg" % path, "data/%s_2.jpg" % path)
    collection = args[1]

    # Get options
    size = (options.width, options.height)
    n = options.testset_size
    location = options.location
    scale = options.scale

    # Get homography
    homography = getSavedHomography(paths)

    # Create testset
    makeTestset(paths, homography, collection, n, location, size, scale)


if __name__ == '__main__':
    try:
        start_time = time.time()
        parser = optparse.OptionParser(formatter=optparse.TitledHelpFormatter(), usage=globals()['__doc__'], version='$Id$')
        parser.add_option ('-v', '--verbose', action='store_true', default=False, help='verbose output')
        parser.add_option ('-l', '--location', type='string', default=".", help='output directory')
        parser.add_option ('-x', '--width', type='int', default=250, help='crop width in pixel')
        parser.add_option ('-y', '--height', type='int', default=250, help='crop height in pixel')
        parser.add_option ('-s', '--scale', type='int', default=1.0, help='crop scale')
        parser.add_option ('-n', '--testset_size', type='int', default=100, help='number of pairs in testset')
        (options, args) = parser.parse_args()

        # Test that we have enough arguments
        if len(args) < 1:
            parser.error ('missing image name (without file ending and _1, e.g. for blu_head_1.jpg, use "blu_head"')
        elif len(args) < 2:
            parser.error ('missing testset name')

        # Elapsed time
        if options.verbose: print time.asctime()
        main()
        if options.verbose: print time.asctime()
        if options.verbose: print 'TOTAL TIME IN MINUTES:',
        if options.verbose: print (time.time() - start_time) / 60.0
        sys.exit(0)
    except KeyboardInterrupt, e: # Ctrl-C
        raise e
    except SystemExit, e: # sys.exit()
        raise e
    except Exception, e:
        print 'ERROR, UNEXPECTED EXCEPTION'
        print str(e)
        traceback.print_exc()
        os._exit(1)



class NonExistantPath(Exception) :
    def __init__(self, path, msg = "") :
        self.path = path
        self.msg = msg
