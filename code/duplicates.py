"""
Python module for finding near duplicates in a set of images
using mirror match and local image features

Jonas Toft Arnfred, 2013-05-23
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import mirrormatch
import numpy
import pylab
import os
import math


####################################
#                                  #
#           Functions              #
#                                  #
####################################


def get_files(folder, verbose = False) :
    files = [path for path in os.listdir(folder) if path[-3:] in ["jpg", "png"]]
    paths = [folder + p for p in files]
    if verbose: 
        for i,f in enumerate(files) :
            print("%2i: %s" % (i,f))
    return files, paths



def get_matches(paths, options = {}) :

    # Override defaults with options
    defaults = {
           "leaf_size": 10,
           "keypoint_type" : "SIFT",
           "descriptor_type" : "SIFT",
           "only_mirror" : True,
           "verbose" : False,
           "max_kp" : 1000,
           "feature_type" : 'L',
    }
    defaults.update(options)

    return mirrormatch.match(paths, defaults)


def get_scores(match_fun, files, threshold = 1.0) :
    nb_images = len(files)
    scores = numpy.zeros((nb_images, nb_images))
    counts = numpy.zeros((nb_images, nb_images))
    matches = match_fun(1.0)
    for p,s,u,(i,j) in zip(*matches) :
        if u < threshold :
            scores[i,j] = scores[i,j] + (1/(u / threshold)) - 1
            scores[j,i] = scores[j,i] + (1/(u / threshold)) - 1
            counts[i,j] = counts[i,j] + 1
            counts[j,i] = counts[j,i] + 1

    scores_sorted, labels = sort_scores(scores, files)
    counts_sorted, labels = sort_scores(counts, files)
    return scores_sorted, counts_sorted, labels


def sigmoid(x):
  return 1 / (1 + math.exp(-x))



def cap(scores, nb_sd = 0.5) :

    def scale(n, median, sd) :
        return (n - median) / (nb_sd * sd)
    
    global_median = numpy.median(scores)
    global_mean = numpy.mean(scores)
    global_sd = numpy.std(scores)
    n_zero = sigmoid(scale(0, global_mean, global_sd))

    # Cap globally
    capped_global = numpy.zeros(scores.shape)
    for i in range(scores.shape[0]) :
        for j in range(scores.shape[1]) :
            capped_global[i,j] = (1 / (1 - n_zero)) * (sigmoid(scale(scores[i,j], global_mean, global_sd)) - n_zero)
    
    return capped_global



def norm_counts(scores, modifier = 1000) :
    """ Normalize according to how many matches there are reported for a given image pair """
    counts = numpy.zeros(scores.shape[0])
    for i in range(scores.shape[0]) :
        counts[i] = sum(scores[i]) / modifier

    scores_norm = numpy.zeros(scores.shape)
    for i in range(scores.shape[0]) :
        for j in range(scores.shape[1]) :
            c = numpy.min((counts[i], counts[j]))
            scores_norm[i,j] = scores[i,j] / (c)
    return scores_norm



def get_F1(result, ground_truth) :
    sTP = numpy.sum(numpy.min((ground_truth, result), axis = 0))
    sTN = numpy.sum(numpy.min((1 - ground_truth, 1- result), axis = 0))
    sFP = numpy.sum(numpy.max((result - ground_truth,  numpy.zeros(result.shape)), axis = 0))
    sFN = numpy.sum(numpy.max((ground_truth - result,  numpy.zeros(result.shape)), axis = 0))
    sPrecision = sTP / (sTP + sFP)
    sRecall = sTP / (sTP + sFN)
    sF1 = 2 * sPrecision * sRecall / (sPrecision + sRecall)
    print("Precision: %.4f" % sPrecision)
    print("Recall: %.4f" % sRecall)
    print("F1: %.4f" % sF1)



def heatmap(scores, labels = None, size = (12,12), cmap = pylab.cm.jet) :
    
    fig, ax = pylab.subplots(figsize=size)
    
    # For colormap: http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
    #heatmap = ax.pcolor(scores_sorted, cmap=cm.Blues)
    heat = ax.pcolor(scores, cmap=cmap)

    # put the major ticks at the middle of each cell
    if labels != None and scores.shape[0] < size[0]*3 : 
        ax.set_xticks(numpy.arange(scores.shape[0])+0.5, minor=False)
        ax.set_yticks(numpy.arange(scores.shape[1])+0.5, minor=False)
        ax.set_xticklabels(labels, minor=False)
        ax.set_yticklabels(labels, minor=False)

    pylab.xlim(0,scores.shape[0])
    pylab.ylim(scores.shape[0],0)



def sort_scores(scores, files) :
    # Get labels
    fname = [f[:3] for f in files]
    rows = sorted(enumerate(fname), key=lambda e : e[1])
    indices, labels = zip(*rows)
    idx = numpy.array(indices)
    
    return scores[idx][:,idx], labels



def normalize_matches(scores, modifier = 1000) :
    """ Normalize each cell in the matrix according to the values of the
    corresponding row and column """
    scores_norm = numpy.zeros(scores.shape)
    for i in range(scores.shape[0]) :
        for j in range(scores.shape[1]) :
            n1, n2 = (scores[i], scores.T[j])
            scores_norm[i,j] = modifier*scores[i,j] / (numpy.sum(n1) + numpy.sum(n2))
    return scores_norm



def norm_scores(scores, amount = 5, verbose = False) :
    s_mean = numpy.mean(scores[scores > 0])
    s_sd = numpy.std(scores[scores > 0])

    # Gather top 5 of every row
    means = numpy.sort(scores, axis = 1)[:, -1*amount:]
    if verbose :
        print(means.shape)
        print(means[0])
        #print(numpy.mean(means, axis=1))
        print(s_mean)
        print(s_sd)

    scores_norm = numpy.zeros(scores.shape)
    for i in range(scores.shape[0]) :
        for j in range(scores.shape[1]) :
            n_mean = numpy.mean(numpy.concatenate((means[i], means[j])))
            n_mean = numpy.min((numpy.mean(means[i]), numpy.mean(means[j])))
            if n_mean < (s_mean) :# + s_sd) :
                s = scores[i, j] / (s_mean)# + s_sd)
            else :
                s = scores[i, j] / n_mean
            if s > 1 :
                scores_norm[i, j] = 1
            elif s < 0 :
                scores_norm[i, j] = 0
            else :
                scores_norm[i, j] = s
    return scores_norm


def gradient(scores) :
    gradients = numpy.zeros(scores.shape)
    scores_sort = numpy.zeros(scores.shape)
    for i in range(scores.shape[0]) :
        line = numpy.sort(scores[i])
        gradients[i] = numpy.gradient(line)[::-1]
        scores_sort[i] = line[::-1]
    return scores_sort, gradients


def trim(scores, margins = (0.0, 1.0), verbose = False) :
    s_max = numpy.mean(numpy.max(scores, axis = 0))
    cut_low = margins[0]*s_max
    cut_high = margins[1]*s_max
    if verbose : 
        print("Cut: (%.2f, %.2f)" % (cut_low, cut_high))
    scores_trim = (scores - cut_low) / (cut_high - cut_low)
    scores_trim[scores_trim < 0] = 0
    scores_trim[scores_trim > 1] = 1
    return scores_trim



def set_diag(scores, k) :
    scores_d = scores.copy()
    for i in range(scores.shape[0]) :
        scores_d[i,i] = k
    return scores_d



def normalize_scores(scores, score_bar = 1, order = 2, max_dist = 999, diagonal = 1.0, cap_multiplier = (0.5, 2.0), verbose = False) :
    
    scores_norm = numpy.zeros(scores.shape)
    min_cap = cap_multiplier[0] * (numpy.mean(scores) / numpy.max(scores))
    max_cap = cap_multiplier[1] * (numpy.mean(scores) + numpy.std(scores)) / numpy.max(scores)
    median = numpy.median(scores)

    if verbose : print("min: %.2f, max: %.2f" % (min_cap, max_cap))

    # normalize according to row and col
    for i in range(scores.shape[0]) :
        for j in range(scores.shape[1]) :
            n1, n2 = (scores[i], scores.T[j])
            neighbors = numpy.concatenate((n1, n2))
            if abs(i - j) > max_dist :
                scores_norm[i, j] = 0
            elif i == j :
                scores_norm[i, j] = diagonal
            elif scores[i, j] < score_bar :
                scores_norm[i, j] = 0
            else :
                # Get all values from cols and rows
                neighbors_sorted = list(reversed(neighbors))
                n_mean = numpy.mean(neighbors_sorted[0:5])

                # Calculate normalized value according to cut
                #s = scores[i, j] / neighbors[-1*order]
                s = scores[i, j] / n_mean
                s_norm = (s - min_cap) / (max_cap - min_cap)
                if s_norm <= 0 :
                    scores_norm[i, j] = 0
                elif s_norm >= 1 :
                    scores_norm[i, j] = 1
                else :
                    scores_norm[i, j] = s_norm
                    
    return scores_norm


def print_scores(scores, files, use_float = False) : 
    
    # Get labels
    labels = [f[:3] for f in files]
    rows = sorted(enumerate(labels), key=lambda e : e[1])
    
    # Top row
    print("      "),
    for i,f in rows :
        print("%s " % f),
    print("\n    +%s-" % ("-----" * scores.shape[1]))
    
    
    # Fields
    for i,f in rows :
        print("%s |" % f),
        for j,_ in rows :
            if scores[i,j] > 0.01 :
                if use_float :
                    print("%1.2f" % scores[i,j]),
                else :
                    print("%4.0f" % scores[i,j]),
            else :
                print("    "),
        print("")
