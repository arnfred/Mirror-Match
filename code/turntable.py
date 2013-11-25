"""
Python module for dealing with the 3D objects dataset published here:
http://www.vision.caltech.edu/pmoreels/Datasets/TurntableObjects/

Jonas Toft Arnfred, 2013-11-18
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import cv2
import numpy
import features
import os
import fnmatch
import display
import mirrormatch
import colors
import random
import pylab

####################################
#                                  #
#           Functions              #
#                                  #
####################################


def evaluate(match_fun, angles, object_type, thresholds, ground_truth_data = None, options = {}) :
    """ Returns number of correct and total matches of match_fun on object:
        match_fun : Function (Function that takes a list of paths and returns matches)
        angles : Int (Rotation in degrees. Must be divisible by 5)
        object_type : String (the object pictured on the turntable)
        thresholds : List[Float] (list of ratio thresholds)
        match_count : List[Int] (List of the amount of possible matches for each feature point)
        options : Dict (Set of parameters for matching etc)
    """

    # Get distance_threshold
    distance_threshold      = options.get("distance_threshold", 5)
    verbose                 = options.get("evaluate_verbose", False)

    # Get paths to the three images
    def get_path(i) : return {
            "A" : get_turntable_path(object_type, angles[0] + i*360, "Bottom"),
            "B" : get_turntable_path(object_type, angles[0] + i*360, "Top"),
            "C" : get_turntable_path(object_type, angles[1] + i*360, "Bottom")
    }

    # Get paths
    def get_matches(i) :
        # Are we weeding out feature that can't be verified by ground truth?
        options["filter_features"] = [] if ground_truth_data == None else ground_truth_data["filter_features"][i]

        # Collect matches
        paths = get_path(i)
        return match_fun([paths["A"], paths["C"]], options = options)(1.0)


    # Get match results
    matches = [get_matches(i) for i in range(3)]
    positions = [m[0] for m in matches]
    ratios = numpy.concatenate([m[2] for m in matches])

    if verbose :
        nb_matches = sum([len(m[0]) for m in matches])
        print("Found %i matches for angles (%i, %i) and object type '%s'" % (nb_matches, angles[0], angles[1], object_type))

    # Get ground trouth
    distances = numpy.concatenate([match_distances(m[0], angles, object_type, distance_threshold) for m in matches])

    if verbose :
        print("Found %i distances" % (len(distances)))

    # Collect precision per ratio_threshold
    def get_count(t, dt) : return len([1 for d,r in zip(distances, ratios) if r <= t and d < dt])
    correct = [get_count(t, distance_threshold) for t in thresholds]
    total = [get_count(t, 9999999) for t in thresholds]
    return { "correct" : correct, "total" : total }



def evaluate_objects(match_fun, angles, object_types, thresholds, ground_truth_data, options = {}) :
    """ Returns number of correct and total matches of match_fun on objects:
        match_fun : Function (Function that takes a list of paths and returns matches)
        angles : (Int, Int) (two angles in degrees. Must be divisible by 5)
        object_types : List[String] (the objects pictured on the turntable)
        thresholds : List[Float] (list of ratio thresholds)
        options : Dict (Set of parameters for matching etc)
    """
    # Get correct and total matches for different thresholds at current angle_increment
    def get_results(object_type) :
        gt = None if ground_truth_data == None else ground_truth_data[object_type]
        return evaluate(match_fun, angles, object_type, thresholds, gt, options)


    results = [get_results(o) for o in object_types]
    correct =  { o : r["correct"] for r,o in zip(results, object_types) }
    total =  { o : r["total"] for r,o in zip(results, object_types) }

    return { "correct" : correct, "total" : total }



def get_turntable_path(object_type, angle, camera_position, turntable_dir = "../../turntable") :
    """ Returns path to image of image_set and object_type where:
        object_type : String (the object pictured on the turntable)
        angle : Int (Rotation in degrees. Must be divisible by 5)
        camera_position : String ("Top" or "Bottom")
        turntable_dir : String (path to folder with image sets)
    """
    image_set = get_image_set(object_type)
    directory = "%s/imageset_%i/%s/%s" % (turntable_dir, image_set, object_type, camera_position)
    file_names = os.listdir(directory)
    file_glob = "img_1-%03i_*_0.JPG" % angle
    image_paths = sorted(fnmatch.filter(file_names, file_glob))
    return "%s/%s" % (directory, image_paths[0])



def load_calibration_image(object_type, angle, camera_position, pattern_position, scale = None, turntable_dir = "../../turntable") :
    """ Loads a checkerboard pattern image taken from the same angle and rotation:
        angle : Int (Rotation in degrees. Must be divisible by 5)
        pattern_position : String (either: "flat", "angled" or "steep")
        scale : Float (the calibration images are double size, so put 2.0)
        turntable_dir : String (path to folder with image sets)
    """
    image_set = get_image_set(object_type)
    directory = "%s/imageset_%i/%s/%s" % (turntable_dir, image_set, object_type, camera_position)
    pattern_index = { "flat" : 0, "steep" : 1, "angled" : 2 }[pattern_position]
    image_index = (angle / 5) * 3 + 1 + pattern_index
    image_path = "%s/imageset_%i/%s/%s/calib%i.jpg" % (turntable_dir, image_set, "Calibration", camera_position, image_index)
    img = cv2.imread(image_path)
    if not scale == None :
        img_scaled = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)))
        return img_scaled
    else :
        return img



# Get calibration points
def get_calibration_points(object_type, angle, camera_position, pattern_position) :
    """ Returns the calibration points from an image with the checkerboard pattern """

    # Load img
    img = load_calibration_image(object_type, angle, camera_position, pattern_position)
    
    if pattern_position == "steep" :
        grid_size = (9, 13)
    else :
        grid_size = (13, 9)
    success, cv_points = cv2.findChessboardCorners(img, grid_size, flags = cv2.CALIB_CB_FILTER_QUADS)
    if not success :
        raise Exception("Can't get lock on chess pattern: Angle: %i, object: %s, camera position: %s, pattern_position: %s" % (angle, object_type, camera_position, pattern_position))
    points = numpy.array([[p[0][0], p[0][1]] for p in cv_points])
    #print(points)
    return points# Get calibration points



def get_foundamental_matrix(object_type, angles, camera_position, scale = 2.0, return_points = False) :
    """ Returns fundamental matrix:
        angles : (Int, Int) (two angles in degrees. Must be divisible by 5)
        camera_position : String (either "Top" or "Bottom")
        scale : Float (scale of images compared to calibration images)
        return_points : Boolean (set to True if you need the calibration points)
    """
    
    # Get image set:
    image_set = get_image_set(object_type)


    # Fetch all images for the angle pair
    points1_flat = get_calibration_points(object_type, angles[0], camera_position[0], "flat")
    points2_flat = get_calibration_points(object_type, angles[1], camera_position[1], "flat")
    points1_angled = get_calibration_points(object_type, angles[0], camera_position[0], "angled")
    points2_angled = get_calibration_points(object_type, angles[1], camera_position[1], "angled")
    points1_steep = get_calibration_points(object_type, angles[0], camera_position[0], "steep")
    points2_steep = get_calibration_points(object_type, angles[1], camera_position[1], "steep")
    
    # Concatenate point sets
    points1 = numpy.concatenate((points1_flat, points1_angled, points1_steep)) / scale
    points2 = numpy.concatenate((points2_flat, points2_angled, points2_steep)) / scale
    
    # Find fundamental matrix based on points
    F, inliers = cv2.findFundamentalMat(points1, points2, method = cv2.FM_RANSAC)
    
    # return matrix with or without points
    if return_points :
        return F, (points1_flat / scale, points2_flat / scale)
    else :
        return F



def epilines(img, points, lines, size = (12, 12)) :
    """ Draws a set of epilines and points on an image """
    # Generate figure
    fig = pylab.figure(figsize=size)
    pylab.imshow(img)
    
    # Get x values
    min_x = 0
    max_x = img.shape[1]
    min_y = 0
    max_y = img.shape[0]
    
    # Limit plot
    pylab.xlim(0,max_x)
    pylab.ylim(max_y-1,0)
    
    # plot lines
    for l, c in zip(lines, colors.get()) :
        # get line functions
        line_fun = lambda x : (-1 * l[0] * x - l[2]) / (float(l[1]))
        # plot line
        pylab.plot([0, max_x], [line_fun(0), line_fun(max_x)], color=c, marker='_')
    
    # plot points
    for p, c in zip(points, colors.get()) :
        # Plot feature match point
        pylab.plot(p[0], p[1], color=c, marker='o')   



def match_distances(matches, angles, object_type, check_threshold) :
    """ Find the distance of matches as measured against two intersection epipolar lines:
        matches : List[(Pos,Pos)] (List of corresponding coordinates in two images)
        angles : (Int, Int) (two angles in degrees. Must be divisible by 5)
        object_type : String (The type of 3d model we are looking at)
        check_threshold : Float (The threshold for a correct correspondence)
    """
    
    # Get features in B
    path_B = get_turntable_path(object_type, angles[0], "Top")
    img_B = features.loadImage(path_B)
    
    # Find fundamental matrices
    F_AC = get_foundamental_matrix(object_type, (angles[0], angles[1]), ("Bottom", "Bottom"), scale = 2.0) # Reference view
    F_AB = get_foundamental_matrix(object_type, (angles[0], angles[0]), ("Bottom", "Top"), scale = 2.0) # Test view
    F_BC = get_foundamental_matrix(object_type, (angles[0], angles[1]), ("Top", "Bottom"), scale = 2.0) # Auxiliary view
    
    # return distances
    return list(calc_match_distances(matches, img_B, F_AB, F_AC, F_BC, check_threshold))



def calc_match_distances(matches, img_B, F_AB, F_AC, F_BC, check_threshold) :
    """ Helper function for match_distances()
        Check ground truth for a set of matches given fundamental matrices,
        As proposed in: "Evaluation of Features Detectors and Descriptors 
        based on 3D objects by Pierre Moreels and Peitro Perona.
    """
    
    # Calculate distance between line and 2D-point
    def dist(line, point) :
        return numpy.abs(line.dot([point[0], point[1], 1]))
    
    # return epipolar line
    def get_lines(points, F) :
        return [l[0] for l in cv2.computeCorrespondEpilines(points.reshape(-1, 1, 2), 1, F)]
    
    # Find points that are on l_AB
    points_A = numpy.array([m[0] for m in matches], dtype=numpy.float32)
    points_C = numpy.array([m[1] for m in matches], dtype=numpy.float32)
    lines_AB = get_lines(points_A, F_AB)
    lines_AC = get_lines(points_A, F_AC)
    
    # Collect features from B, so we can check the match there
    feature_pos_B = features.getPositions(features.getKeypoints(img_B))

    for p_A, p_C, l_AB, l_AC in zip(points_A, points_C, lines_AB, lines_AC) :
        
        # Is p_C on l_AC?
        min_dist = dist(l_AC, p_C)
        if min_dist < check_threshold :
            
            # Collect all features in B that are on l_AB:
            points_B = numpy.array([p_B for p_B in feature_pos_B if dist(l_AB, p_B) < check_threshold], dtype=numpy.float32)
            
            # Check if points match up
            if len(points_B) > 0 :
                lines_BC = get_lines(points_B, F_BC)
                min_dist = max((numpy.min([dist(l_BC, p_C) for l_BC in lines_BC]), min_dist))
                
        yield min_dist


def get_image_set(object_type) :
    sets = {
        1 : ["Conch",  "FlowerLamp",        "Motorcycle",  "Rock", "Bannanas",  "Car",          "Desk",   "GrandfatherClock",  "Robot",       "TeddyBear", "Base",      "Car2",         "Dog",    "Horse", "Tricycle"],
        2 : ["Clock",  "EthernetHub",  "Hicama",  "Pepper"],
        3 : ["Dremel",  "JackStand",  "Sander",  "SlinkyMonster",  "SprayCan"],
        4 : ["FireExtinguisher",  "Frame",  "Hat", "StaplerRx"],
        5 : ["Carton",  "Clamp",  "EggPlant",  "Lamp",  "Mouse",  "Oil"],
        6 : ["Basket",  "Clipper",  "CupSticks",  "Filter",  "Mug",  "Shoe"],
        7 : ["Pops", "Speaker", "BoxingGlove",  "CollectorCup", "Utilities"],
        8 : ["Camera",  "DishSoap",  "Nesquik",  "PotatoChips",   "Sponge"],
        9 : ["Camera2",     "Cloth",     "FloppyBox", "CementBase",  "Dinosaur",  "PaperBin"],
        10 : ["Phone2",      "RollerBlade",  "Tripod",  "RiceCooker",  "Spoons",       "VolleyBall"],
        11 : ["Gelsole",   "MouthGuard",  "Razor",       "Toothpaste", "Abroller",  "DVD",          "Keyboard",  "PS2"],
        12 : ["Bush",  "DuckBank",  "Eggs",  "Frog",  "LightSaber"],
        13 : ["Coffee",  "LavaBase",  "SwanBank",  "ToolBox",  "Vegetta"],
        14 : ["BoxStuff",     "Standing", "BallSander", "StorageBin"],
        15 : ["Globe",  "Pineapple"]
    }
    for key,value in sets.iteritems() :
        if object_type in value :
            return key

    raise Exception("No object matching object_type of '%s'" % object_type)



def ground_truth(angles, object_type, options = {}) :
    """ Find the amount of total possible correspondences for all lightning conditions. 
        angles : (Int, Int) (two angles in degrees. Must be divisible by 5)
        object_type : String (The type of 3d model we are looking at)
        return_matches : Boolean (set to True to return the correspondences found too)
        options : Dict (Set of parameters for matching etc)
    """
    verbose = options.get("evaluate_verbose", False)
    nb_correspondences = 0
    filter_features = []
    for i in range(3) :
        curr_matches = list(calc_ground_truth(angles, object_type, lightning_index = i, return_matches = False, options = options))
        # Create list of features that should be kept
        curr_ff = [i for i, m in enumerate(curr_matches) if len(m) == 0]
        nb_correspondences += len(curr_matches) - len(curr_ff)
        filter_features.append(curr_ff)

    if verbose :
        print("There are %i theoretically possible correspondences for object '%s' at angles (%i,%i)" % (nb_correspondences, object_type, angles[0], angles[1]))
     
    return { "nb_correspondences" : nb_correspondences, "filter_features" : filter_features }


# Load keypoints
def keypoints(object_type, angle, viewpoint) :
    path = get_turntable_path(object_type, angle, viewpoint)
    points = features.getPositions(features.getFeatures([path])[1])
    return numpy.array(points, dtype=numpy.float32)

# return epipolar line
def get_lines(points, F) :
    return [l[0] for l in cv2.computeCorrespondEpilines(points.reshape(-1, 1, 2), 1, F)]

# Calculate distance between line and 2D-point
def dist(line, point) :
    return numpy.abs(line.dot([point[0], point[1], 1]))



def calc_ground_truth(angles, object_type, lightning_index = 0, return_matches = False, options = {}) :
    """ Find the amount of total possible correspondences. 
        For each feature in A, check if there is a feature in C such that the 
        epipolar constraints for a correct match are fulfilled for any point in B:
        angles : (Int, Int) (two angles in degrees. Must be divisible by 5)
        object_type : String (The type of 3d model we are looking at)
        return_matches : Boolean (set to True to return the correspondences found too)
        options : Dict (Set of parameters for matching etc)
    """
    
# Get distance_threshold
    distance_threshold  = options.get("distance_threshold", 5)
    verbose             = options.get("verbose", False)

    # Get paths to the three images
    keypoints_A = keypoints(object_type, angles[0]+360*lightning_index, "Bottom")
    keypoints_B = keypoints(object_type, angles[0]+360*lightning_index, "Top")
    keypoints_C = keypoints(object_type, angles[1]+360*lightning_index, "Bottom")

    # Find fundamental matrices
    F_AC = get_foundamental_matrix(object_type, (angles[0], angles[1]), ("Bottom", "Bottom"), scale = 2.0)
    F_AB = get_foundamental_matrix(object_type, (angles[0], angles[0]), ("Bottom", "Top"), scale = 2.0)
    F_BC = get_foundamental_matrix(object_type, (angles[0], angles[1]), ("Top", "Bottom"), scale = 2.0)

    # For every point in A find the corresponding lines in B and C
    lines_AB = get_lines(keypoints_A, F_AB)
    lines_AC = get_lines(keypoints_A, F_AC)

    # For every epiline in B and C corresponding to a point in A
    for i, (p_A, l_AB, l_AC) in enumerate(zip(keypoints_A, lines_AB, lines_AC)) :

        # Find all points on the line in B and C
        points_B = numpy.array([p_B for p_B in keypoints_B if dist(l_AB, p_B) < distance_threshold], dtype=numpy.float32)
        points_C = numpy.array([p_C for p_C in keypoints_C if dist(l_AC, p_C) < distance_threshold], dtype=numpy.float32)

        # For every point in B on l_AB, see if there is a point in C on l_AC that lies on the epipolar line of p_B in image C: l_BC
        if len(points_B) > 0 and len(points_C) > 0 :

            # Get distances from every point in C on line l_AC to every line in C corresponding to a point in B on line l_AB
            get_distances = lambda line : [(dist(line, p_C), p_C) for p_C in points_C]
            distances = features.flatten([get_distances(l_BC) for l_BC in get_lines(points_B, F_BC)])

            # Count how many potential matches there are
            yield [(p_A, p_C) for d, p_C in distances if d < distance_threshold]

        else :
            yield []
