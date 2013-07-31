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
import graph_tool.all as gt
import math
import louvain
import colors
import pylab
from scipy.misc import imresize, imread
import Image
from itertools import combinations, groupby, tee, product, combinations_with_replacement, dropwhile
from sklearn import metrics


####################################
#                                  #
#             Images               #
#                                  #
####################################

def appendimages(im1, im2, seperator = 0) :
    """ return a new image that appends the two images side-by-side.
    """

    barrier = numpy.ones((im1.shape[0],seperator)) * 255

    if (im1.shape[0] == im2.shape[0]) :
        tmp = im2

    elif (im1.shape[0] > im2.shape[0]) :
        tmp = numpy.ones((im1.shape[0], im2.shape[1])) * 255
        tmp[0:im2.shape[0], :] = im2

    elif (im1.shape[0] < im2.shape[0]) :
        tmp = numpy.ones((im1.shape[0], im2.shape[1])) * 255
        tmp[0:im1.shape[0], :] = im2[0:im1.shape[0],:]

    else :
        print("Detonating thermo-nuclear devices")

    return numpy.concatenate((im1,barrier, tmp), axis=1)





def keypoints(im, pos) :
    """ show image with features. input: im (image as array), 
        locs (row, col, scale, orientation of each feature) 
    """
    # Plot all keypoints
    pylab.gray()
    pylab.imshow(im)
    pylab.plot([p[1] for p in pos], [p[0] for p in pos], '.b')
    pylab.axis('off')
    pylab.show()


def compareKeypoints(im1, im2, pos1, pos2, filename = None, separation = 0) :
    """ Show two images next to each other with the keypoints marked
    """

    # Construct unified image
    im3 = appendimages(im1,im2, separation)

    # Find the offset and add it
    offset = im1.shape[1]
    pos2_o = [(x+offset + separation,y) for (x,y) in pos2]

    # Create figure
    fig = pylab.figure(frameon=False, figsize=(10.0, 7.0))
    #ax = pylab.Axes(fig, [0., 0., 1., 1.])

    # Show images
    pylab.gray()
    pylab.imshow(im3)
    pylab.plot([x for x,y in pos1], [y for x,y in pos1], marker='o', color = '#00aaff', lw=0)
    pylab.plot([x for x,y in pos2_o], [y for x,y in pos2_o], marker='o', color = '#00aaff', lw=0)
    pylab.axis('off')

    pylab.xlim(0,im3.shape[1])
    pylab.ylim(im3.shape[0],0)

    if filename != None :
        fig.savefig(filename, bbox_inches='tight', dpi=72)


def matchPoints(im1, im2, matches, dist = None, filename = None, max_dist = 100, matches_im1 = None, dist_im1 = None, matches_im2 = None, dist_im2 = None) :
    """ show a figure with lines joining the accepted matches in im1 and im2
        input: im1,im2 (images as arrays), locs1,locs2 (location of features), 
        matchscores (as output from 'match'). 
    """

    separation = 20

    # Construct unified image
    im3 = appendimages(im1,im2, separation)

    # Create figure
    fig = pylab.figure(frameon=False, figsize=(12.0, 8.0))
    ax = pylab.Axes(fig, [0., 0., 1., 1.])

    ax.set_axis_off()
    fig.add_axes(ax)

    # Display image
    pylab.gray()
    ax.imshow(im3)

    # Get colors
    if dist != None and len(dist) == len(matches) :
        cs = [colors.getRedGreen(numpy.log(d+1)/numpy.log(max_dist)) for d in dist]
    else :
        cs = ['#00aaff' for m in matches]

    # Get colors for images
    if dist_im1 != None and len(dist_im1) == len(matches_im1) :
        cs_im1 = [colors.getRedGreen(numpy.log(d+1)/numpy.log(max_dist)) for d in dist_im1]
    else :
        cs_im1 = ['#00aaff' for m in matches]
    if dist_im2 != None and len(dist_im2) == len(matches_im2) :
        cs_im2 = [colors.getRedGreen(numpy.log(d+1)/numpy.log(max_dist)) for d in dist_im2]
    else :
        cs_im2 = ['#00aaff' for m in matches]
    
    # Plot all lines
    offset_x = im1.shape[1]
    for i,((x1,y1),(x2,y2)) in enumerate(matches) :
        ax.plot([x1, x2+offset_x + separation], [y1,y2], color=cs[i], lw=0.8)
    if matches_im1 != None :
        for i,((x1,y1),(x2,y2)) in enumerate(matches_im1) :
            ax.plot([x1, x2], [y1,y2], color=cs_im1[i], lw=0.8)
    if matches_im2 != None :
        for i,((x1,y1),(x2,y2)) in enumerate(matches_im2) :
            ax.plot([x1+offset_x + separation, x2+offset_x + separation], [y1,y2], color=cs_im2[i], lw=0.8)

    pylab.xlim(0,im3.shape[1])
    pylab.ylim(im3.shape[0],0)

    if filename != None :
        fig.savefig(filename, bbox_inches='tight', dpi=72)



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
    matchPoints(images[0], images[1], keypoints[0], keypoints[1], masked_matchPos)



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


def scoreNormHist(resultMat, labels) :
    # Normalize resultMat row by row with respect to mean and standard deviation
    resultMat_norm = numpy.zeros(resultMat.shape)
    label_array = numpy.array(labels)
    for i,(row,l) in enumerate(zip(resultMat, labels)) :

        same = label_array == l
        diff = label_array != l
        mean_same = numpy.mean(row[same])
        mean_diff = numpy.mean(row[diff])
        weighted_mean = mean_same * (numpy.sum(same) / float(len(labels))) + mean_diff * (numpy.sum(diff) / float(len(labels)))
        sd = numpy.sqrt(numpy.var(row))
        ratio = mean_same / (mean_diff + 0.000001)
        resultMat_norm[i] = (row - weighted_mean) / (sd + 00000.1)
        
    # Make sure the diagonal is zero
    for i in range(len(labels)) : resultMat_norm[i][i] = 0.0
        
    # Create mask
    resultMask = numpy.zeros(resultMat.shape, dtype=numpy.bool)
    for i,l in enumerate(labels) :
        resultMask[i] = label_array == l

    # Get vectors with results for same and diff
    same = resultMat_norm[resultMask]
    diff = resultMat_norm[numpy.invert(resultMask)]

    mean_same = numpy.mean(same)
    mean_diff = numpy.mean(diff)
    variance = numpy.mean([numpy.var(same), numpy.var(diff)])
    sd = numpy.sqrt(variance)

    print("Mean(same):\t\t{0:.3f}".format(mean_same))
    print("Mean(diff):\t\t{0:.3f}".format(mean_diff))
    print("Diff of means:\t\t{0:.3f}".format(numpy.abs(mean_diff - mean_same)))
    print("Standard deviation:\t{0:.3f}".format(sd)) 
    print("# of sd's:\t\t{0:.3f}".format(numpy.abs(mean_diff - mean_same)/sd))

    x_min = min([numpy.min(same),numpy.min(diff)]) 
    x_max = max([numpy.max(same),numpy.max(diff)]) 
    margin = (x_max - x_min) * 0.2

    pylab.subplot(1,2,1)
    pylab.hist(same, bins=20, label="Same", color="green", alpha=0.65)
    pylab.legend()
    pylab.xlim(x_min - margin ,x_max + margin )
    removeDecoration()

    pylab.subplot(1,2,2)
    pylab.hist(diff, bins=20,label="Diff", color="red", alpha=0.65)
    pylab.legend()
    pylab.xlim(x_min - margin ,x_max + margin )
    removeDecoration()


def accuHist(accu_list, labels, colors = ["blue", "cyan", "green", "orange", "red"], ylim = 100) :
    n = len(accu_list)
    pylab.figure(figsize=(10, 3))
    for i,(a,l,c) in enumerate(zip(accu_list, labels, colors)) :
        pylab.subplot(1,n,(i+1))
        pylab.hist(a, bins=20, label=l, color=c, alpha=0.65)
        pylab.legend()
        pylab.xlim(0,1.01)
        pylab.ylim(0,ylim)
        removeDecoration()


def accuDetail(correct, total, legend, ylim = 100, treshold=1000) :
    get_index = lambda cs : list(dropwhile(lambda (i,c) : sum(c) < treshold,enumerate(cs)))[0][0]
    indices = [get_index(cs) for cs in correct]
    print(indices)
    
    for c,t,l,i in zip(correct, total, legend, indices) :
        print("%s:\t%i of %i\t(%.2f%%)" % (l, sum(c[i]), sum(t[i]), 100*sum(c[i])/float(sum(t[i]))))
        
    get_accu = lambda ts,cs,index : [1 if t == 0 else c/float(t) for t,c in zip(ts[index], cs[index])]
    accu = [get_accu(ts, cs, index) for ts,cs,index in zip(total, correct, indices)]
    accuHist(accu, legend, ylim=ylim)


def recallPlot(correct, total, nb_correspondences, legends, colors = ["blue", "red", "green", "orange", "cyan"], ylim=(0.0,1.01), xlim = None, size = (4,6)) :
    fig = pylab.figure(figsize=size)
    ax = pylab.subplot(111)
    for ts, cs, l, color in zip(total, correct, legends, colors) :
        xs = [(sum(t) - sum(c))/float(sum(t)) for c, t in zip(cs,ts)]
        ys = [sum(c)/float(nb_correspondences) for c in cs]
        pylab.plot(xs, ys, '-', label=l, color=color, alpha=0.95)
        pylab.legend(loc="best")
    removeDecoration()
    pylab.xlabel("1 - precision")
    pylab.ylabel("# correct / %i" % nb_correspondences)
    pylab.ylim(ylim[0],ylim[1])
    if xlim != None : pylab.xlim(0,xlim)

def accuPlot(correct, total, legends, colors = ["blue", "red", "green", "orange", "cyan"], ylim=(0.0,1.01), xlim = None, size = (4,6), compareCorrect = None, compareTotal = None, compareLegend = None, outside = False) :
    fig = pylab.figure(figsize=size)
    ax = pylab.subplot(111)
    for ts,cs,l,color in zip(total, correct, legends, colors) :
        xs = [sum(t) for t in ts]
        ys = [1 if sum(t) == 0 else sum(c)/float(sum(t)) for (c, t) in zip(cs, ts)]
        pylab.plot(xs, ys, '-', label=l, color=color, alpha=0.95)
        pylab.legend(loc="best")
    if compareCorrect != None and compareTotal != None and compareLegend != None :
        for ts,cs,l,color in zip(compareTotal, compareCorrect, compareLegend, ["blue", "red", "green", "orange", "red"]) :
            xs = [sum(t) for t in ts]
            ys = [1 if sum(t) == 0 else sum(c)/float(sum(t)) for (c, t) in zip(cs, ts)]
            pylab.plot(xs, ys, '--', label="%s" % (l), color=color, alpha=1)
            pylab.legend(loc="best")
    removeDecoration()
    pylab.xlabel("# of Matches")
    pylab.ylabel("Accuracy")
    pylab.ylim(ylim[0],ylim[1])

    if outside == True :
        # Shink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if xlim != None : pylab.xlim(0,xlim)


def distHist(dist, dist_treshold = 5, dist_distinct = None, accuracity = None, accu_y_lim = 100) :

    plots = 2 if accuracity == None else 3
    dist_under_median = [d for d in dist if d <= (numpy.median(dist) * 2)]
    dist_under_treshold = [d for d in dist if d <= dist_treshold]
    if (len(dist) == 0) : return
    dist_p = float(len(dist_under_treshold)) / float(len(dist))
    if dist_distinct != None :
        distinct_under_treshold = [d for d in dist_distinct if d <= dist_treshold]
        distinct_p = float(len(distinct_under_treshold)) / float(len(dist_distinct))

    pylab.figure(figsize=(10, 3))

    pylab.subplot(1,plots,1)
    pylab.hist(dist, bins=20, label="Distances", color="blue", alpha=0.65)
    pylab.legend()
    removeDecoration()

    pylab.subplot(1,plots,2)
    pylab.hist(dist_under_median, bins=20, label="Distances", color="cyan", alpha=0.65)
    pylab.legend()
    removeDecoration()

    if accuracity != None :
        pylab.subplot(1,plots,3)
        pylab.hist(accuracity, bins=20, label="Accuracity", color="green", alpha=0.65)
        pylab.legend()
        pylab.xlim(0,1.01)
        pylab.ylim(0,accu_y_lim)
        removeDecoration()


    if dist_distinct != None :
        print("Number of matches:\t%i\t(distinct: %i)" % (len(dist), len(dist_distinct)))
        print("Correct matches:\t%i\t(distinct: %i)\t\t[under %ipx error]" % (len(dist_under_treshold), len(distinct_under_treshold), dist_treshold) )
        print("Success Rate:\t\t%.2f%%\t(distinct: %.2f%%)" % (dist_p*100, distinct_p*100))
    else :
        print("Number of matches:\t%i" % len(dist))
        print("Correct matches:\t%i\t[under %ipx error]" % (len(dist_under_treshold), dist_treshold) )
        print("Under %ipx error:\t%.2f%%" % (dist_treshold, dist_p*100))



def scoreHist(scores) : 

    if (len(scores[0]) == 3) :
        same = [s for b,s,p in scores if b and not math.isnan(s) ]
        diff = [s for b,s,p in scores if not b and not math.isnan(s)] 
    else :
        same = [s for b,s in scores if b and not math.isnan(s)]
        diff = [s for b,s in scores if not b and not math.isnan(s)] 

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
    removeDecoration()

    pylab.subplot(1,2,2)
    pylab.hist(diff, bins=20,label="Diff", color="red", alpha=0.65)
    pylab.legend()
    pylab.xlim(x_min - margin ,x_max + margin )
    removeDecoration()



def scoreRatioHist(resultMat, labels, lower_is_better=False) :
    scores = numpy.zeros(labels.shape)
    for i,row in enumerate(resultMat) :
        equal_labels = (labels == labels[i])
        diff_labels = (labels != labels[i])
        mean_same = numpy.mean(row[equal_labels]) + 0.0000001
        mean_diff = numpy.mean(row[diff_labels]) + 0.0000001
        ratio = mean_diff/mean_same if lower_is_better else mean_same/mean_diff
        scores[i] = ratio
        
    # Remove outliers
    scores[scores > 3] = 3
    below_1 = numpy.sum(scores<1)
    above_1 = numpy.sum(scores>1)
    equal_1 = numpy.sum(scores==1)
    above_3 = numpy.sum(scores>3)
    if equal_1 > 0 : 
        gray_start = 0.95
        gray_end = 1.05
    else :
        gray_start = 1
        gray_end = 1
    ax = pylab.subplot(1,1,1)
    pylab.hist(scores[scores < 1],bins=numpy.linspace(0,gray_start,7), color="red", alpha=0.7, label="< 1.0 (%i)" % below_1)
    if equal_1 > 0 : pylab.hist(scores[scores == 1],bins=numpy.linspace(gray_start,gray_end,2), color="grey", alpha=0.7, label="= 1.0 (%i)" % equal_1)
    pylab.hist(scores[scores > 1],bins=numpy.linspace(gray_end,3,14), color="green", alpha=0.7, label="> 1.0 (%i)" % above_1)
    pylab.xlim(0,3.01)
    pylab.legend(loc="best")
    #ax.set_xscale("log")
    removeDecoration()


def farPlot(scores, n = 500, lower_is_better = False) :

    if (len(scores[0]) == 3) :
        same = [s for b,s,p in scores if b and not math.isnan(s) ]
        diff = [s for b,s,p in scores if not b and not math.isnan(s)] 
    else :
        same = [s for b,s in scores if b and not math.isnan(s)]
        diff = [s for b,s in scores if not b and not math.isnan(s)] 

    # set compare function
    compare = (lambda a,b : a < b) if lower_is_better else (lambda a,b : a > b)

    # Create linspace
    min_val = numpy.min(same)
    max_val = numpy.max(same)
    start = min_val if lower_is_better else max_val
    end = max_val if lower_is_better else min_val
    tresholds = numpy.linspace(start,end,n)

    # Calculate x and y rows
    sum_same = float(len(same))
    sum_diff = float(len(diff))
    false_positive_rate = [(len([s for s in diff if compare(s,t)]) / sum_diff)*100 for t in tresholds]
    true_positive_rate = [len([s for s in same if compare(s,t)]) / sum_same for t in tresholds]

    # Show figure
    fig = pylab.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(false_positive_rate, true_positive_rate)
    ax.set_xscale('log')
    removeDecoration()



def clusterPlot(resultMat, labels, pruner, ylim=0.5, xlim=8) :
    xs = numpy.linspace(0.001,xlim,100)

    def getRandScore(l) :
        pruned_w = pruner(resultMat, l, n=500, start=0.0)
        p = louvain.cluster(pruned_w)
        #ars = metrics.adjusted_rand_score(labels, p)
        amis = metrics.adjusted_mutual_info_score(labels,p)
        return amis

    ys = [getRandScore(x) for x in xs]
    pylab.plot(xs,ys)
    pylab.ylim(0,ylim)
    removeDecoration()



def showPartitions(points, partitioning, image = None) :
    max_x = numpy.max(points[:,0])
    max_y = numpy.max(points[:,1])
    cs = colors.get()
    pylab.gray()
    if image != None :
        pylab.imshow(image)
        pylab.xlim(0,image.shape[1])
        pylab.ylim(image.shape[0],0)
    else :
        pylab.xlim(0,max_y*1.1)
        pylab.ylim(0,max_x*1.1)

    for pos,p in zip(points, partitioning) :
        pylab.plot(pos[0], pos[1], color=cs[p], marker='o')

    removeDecoration()



####################################
#                                  #
#            Clusters              #
#                                  #
####################################



def trait_graph(tg, images) :
    graph_on_images(
            tg, 
            images, 
            clusters=tg.vp["partitions"], 
            path="trait_graph.png",
            edge_color=tg.ep["edge_colors"], 
            vertex_size=tg.vp["variance"])


def draw_graph(g, clusters="orange", filename="graph.png", size=(1000,1000)) :
    """ Show a graph with its clustering marked
    """
    # Get indices
    indices = g.vertex_properties["indices"]

    # Get class colors
    class_colors = g.vertex_properties["class_colors"]

    # Get weights and positions
    weights = g.edge_properties["weights"]
    pos = gt.sfdp_layout(g, eweight=weights)

    # Print graph to file
    gt.graph_draw(g, pos=pos, output_size=size, vertex_halo=True, vertex_halo_color=class_colors, vertex_color=clusters,
               vertex_fill_color=clusters, vertex_size=5, edge_pen_width=weights, output=filename)

    g_img = imread(filename)
    pylab.imshow(g_img)



def graph_partitions(g, clusters, filename="graph_clusters.png") :
    """ Create an image where the clusters are disconnected
    """
    # Prune inter cluster edges
    intra_cluster = g.new_edge_property("bool")
    intra_cluster.fa = [(clusters[e.source()] == clusters[e.target()]) for e in graph.edges()]

    # Create graph with less edges
    g_cluster = gt.GraphView(g, efilt=intra_cluster)

    graph(g_cluster, clusters, filename=filename)



def graph_on_images(graph, images, clusters = "orange", path="graph_on_images.png", vertex_size=15, edge_color=[0.0, 0.0, 0.0, 0.8]) :
    """ Displays the feature points of the graph as they are located on the images
        Input: graph [Graph]
               images [List of images]
    """

    def tails(it):
        """ tails([1,2,3,4,5]) --> [[1,2,3,4,5], [2,3,4,5], [3,4,5], [4,5], [5], []] """
        while True:
            tail, it = tee(it)
            yield tail
            next(it)

    # Interpolate images to double size
    scale = 2.0
    separation = 20

    # Show in gray-scale
    pylab.gray()

    # Image paths
    bg_path = "graph_background.png"
    fg_path = "graph_foreground.png"
    merge_path = path

    # Put images together and resize
    bg_small = appendimages(images[0], images[1], seperator = separation)
    bg = imresize(bg_small, size=scale, interp='bicubic')
    pylab.imsave(bg_path, bg)

    # Calculate offsets
    offsets = map(sum, [list(t) for t in tails(map(lambda i : (i.shape[1] + separation)*scale, images))])[::-1]
    print(offsets)

    # Get scaled positions
    ind_prop = graph.vertex_properties["indices"]
    x,y = (graph.vertex_properties["x"], graph.vertex_properties["y"])
    pos = graph.new_vertex_property("vector<float>")
    for v in graph.vertices() : 
        x_scaled = x[v] * scale + offsets[ind_prop[v]]
        y_scaled = y[v] * scale
        pos[v] = [x_scaled, y_scaled]

    # Get weights
    weights = graph.edge_properties["weights"]

    # Draw graph
    class_colors = graph.vertex_properties["class_colors"]
    gt.graph_draw(graph, 
                  pos=pos, 
                  fit_view=False, 
                  output_size=[bg.shape[1], bg.shape[0]],
                  vertex_halo=True,
                  vertex_halo_color="black",
                  vertex_size=vertex_size,
                  vertex_fill_color=clusters,
                  edge_color=edge_color,
                  edge_pen_width=weights,
                  output=fg_path
                 )
    
    # Merge the graph and background images
    background = Image.open(bg_path)
    foreground = Image.open(fg_path)
    background.paste(foreground, (0, 0), foreground)
    background.save(merge_path)
    
    # Show resulting image
    im = pylab.imread(merge_path)
    pylab.imshow(im)



def faceGraph(graph, partitions, paths, filename = "facegraph.png") :
    partition_prop = graph.new_vertex_property("int")
    partition_prop.fa = partitions
    path_prop = graph.new_vertex_property("string")
    for v, p in zip(graph.vertices(), paths) : path_prop[v] = p
    pos = gt.sfdp_layout(graph, eweight=graph.ep["weights"])
    gt.graph_draw(graph, 
                  pos=pos, 
                  output_size=(2000, 1400), 
                  vertex_halo=True, 
                  vertex_halo_color=partition_prop, 
                  #vertex_fill_color=partitioning, 
                  vertex_anchor=0,
                  vertex_pen_width=10,
                  vertex_color=partition_prop,
                  vertex_surface=path_prop,
                  vertex_size=50, 
                  edge_pen_width=graph.ep["weights"], 
                  output=filename)

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

