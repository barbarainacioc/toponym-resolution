import math

import healpy
import numpy as np
import tensorflow as tf
from geopy import distance
from scipy import stats
from sklearn.metrics import mean_absolute_error

from nn_model import *


# confidence intervals for the mean error values
def mean_confidence_interval( data ):
    confidence = 0.95
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a) , stats.sem(a)
    h = se * stats.t._ppf( ( 1 + confidence ) / 2.0 , n-1 )
    return m , m-h , m+h , h


# confidence intervals for the median error values
def median_confidence_interval( data ):
    n_boots = 10000
    sample_size = 50
    a = 1.0 * np.array(data)
    me = [ ]
    np.random.seed(seed=0)
    for _ in range(0,n_boots):
        sample = [ a[ np.random.randint( 0 , len(data) - 1 ) ] for _ in range(0,sample_size) ]
        me.append( np.median( sample ) )
    med = np.median(data)
    mph = np.percentile(me, 2.5)
    mmh = np.percentile(me, 97.5)
    return med , mph , mmh


# convert lat and lon to a healpix code encoding a region, with a given resolution
def latlon2healpix( lat , lon , res ):
    lat = lat * math.pi / 180.0
    lon = lon * math.pi / 180.0
    xs = ( math.cos(lat) * math.cos(lon) )
    ys = ( math.cos(lat) * math.sin(lon) )
    zs = ( math.sin(lat) )
    return healpy.vec2pix( int(res) , xs , ys , zs )


# convert healpix code of a given resolution, back into lat and lon coordinates
def healpix2latlon( code , res ):
    [xs, ys, zs] = healpy.pix2vec( int(res) , code )
    lat = float( math.atan2(zs, math.sqrt(xs * xs + ys * ys)) * 180.0 / math.pi )
    lon = float( math.atan2(ys, xs) * 180.0 / math.pi )
    return [ lat , lon ]


# return geodesic distance between two points
def geodistance( coords1 , coords2 ):
    lat1 , lon1 = coords1[ : 2]
    lat2 , lon2 = coords2[ : 2]

    try: return distance.vincenty( ( lat1 , lon1 ) , ( lat2 , lon2 ) ).meters / 1000.0
    except: return distance.great_circle( ( lat1 , lon1 ) , ( lat2 , lon2 ) ).meters / 1000.0


# tensorflow function for computing the great circle distance (geodesic distance) between two points
def geodistance_tensorflow( p1 , p2 ):
    aa0 = p1[:,0] * 0.01745329251994329576924
    aa1 = p1[:,1] * 0.01745329251994329576924
    bb0 = tf.atan2(p2[:,2], K.sqrt(p2[:,0] ** 2 + p2[:,1] ** 2)) * 180.0 / 3.141592653589793238462643383279502884197169399375105820974944592307816406286
    bb1 = tf.atan2(p2[:,1], p2[:,0]) * 180.0 / 3.141592653589793238462643383279502884197169399375105820974944592307816406286
    bb0 = bb0 * 0.01745329251994329576924
    bb1 = bb1 * 0.01745329251994329576924
    sin_lat1 = K.sin( aa0 )
    cos_lat1 = K.cos( aa0 )
    sin_lat2 = K.sin( bb0 )
    cos_lat2 = K.cos( bb0 )
    delta_lng = bb1 - aa1
    cos_delta_lng = K.cos(delta_lng)
    sin_delta_lng = K.sin(delta_lng)
    d = tf.atan2(K.sqrt((cos_lat2 * sin_delta_lng) ** 2 + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2), sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng )
    return K.mean( 6371.0087714 * d , axis = -1 )


def cube(vector):
    new_vector = (vector**3.0)/(K.sum(vector**3.0))
    return new_vector


def normalize_size(words, size):
    if len(words) == size:
        return words, 0
    if len(words) > size:
        return words[:size], 0
    else:
        missing = size-len(words)
        words += [" "]*missing
        return words, missing


def find_sentence_index(sentence, text, current):
    index = text.index(sentence[0], current)
    for i in range(len(sentence)):
        if sentence[i] != text[index+i]:
            return find_sentence_index(sentence, text, index+1)
    return index


def get_text_words(mention, sentence, text, size):
    text_size = len(text)
    if text_size<=size:
        missing = size-text_size
        text += [" "]*missing
        return text, missing
    else:
        start_sent_idx = find_sentence_index(sentence, text, 0)
        try:
            start_mention_idx = text.index(mention[0], start_sent_idx)
        except ValueError:
            matching = [s for s in sentence if mention[0] in s]
            start_mention_idx = text.index(matching[0], start_sent_idx)

        left_side = round(size/2)
        if (start_mention_idx-left_side)>=0:
            words = text[start_mention_idx-left_side:start_mention_idx+len(mention)]
        else:
            words = text[0:start_mention_idx+len(mention)]

        rigth_side = size-len(words)
        if (start_mention_idx+len(mention)+rigth_side) <= text_size:
            words += text[start_mention_idx+len(mention):start_mention_idx+len(mention)+rigth_side]
        else:
            words += text[start_mention_idx+len(mention):text_size]
            if len(words)<size:
                to_add_begin = size-len(words)
                words = text[text_size-len(words)-to_add_begin:text_size-len(words)] + words
        return words, 0


def get_mentions(file_name):
    mentions = []

    with open(file_name, "r") as file:
        for line in file:
            content = line.split("\t")
            mentions.append(content[0])
        file.close()
    return mentions


def get_results(args, test_data, distance_file, results_file, predictions, test_instances_sz, 
                encoder, Y1_test, Y2_test, Y3_test, Y4_test, Y5_test, Y6_test):
    distances = [ ]
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0

    test_names = get_mentions(test_data)

    f = open(distance_file, 'w')

    y_classes = predictions[1].argmax(axis=-1)
    y_classes = encoder.inverse_transform(y_classes)

    acc_161 = 0
    for pos in range(test_instances_sz):

        ( xs, ys, zs ) = predictions[0][pos]
        lat = float( math.atan2(zs, math.sqrt(xs * xs + ys * ys)) * 180.0 / math.pi )
        lon = float( math.atan2(ys, xs) * 180.0 / math.pi )

        dist = geodistance( ( Y1_test[pos][0] , Y1_test[pos][1] ) , ( lat , lon ) )
        if dist <=161:
            acc_161 += 1
        f.write(test_names[pos].replace(" ","_") + "\t" + str(lat) + "\t" + str(lon) + "\t" + str(Y1_test[pos][0]) + "\t" + str(Y1_test[pos][1]) + "\t" + str(dist) + "\n")
        distances.append( dist )
        if latlon2healpix ( Y1_test[pos][0] , Y1_test[pos][1] , math.pow(4 , 5) ) == latlon2healpix ( lat, lon , math.pow(4 , 5) ): correct4 += 1
        if latlon2healpix ( Y1_test[pos][0] , Y1_test[pos][1] , math.pow(4 , 4) ) == latlon2healpix ( lat, lon , math.pow(4 , 4) ): correct3 += 1
        if latlon2healpix ( Y1_test[pos][0] , Y1_test[pos][1] , math.pow(4 , 3) ) == latlon2healpix ( lat, lon , math.pow(4 , 3) ): correct2 += 1
        if latlon2healpix ( Y1_test[pos][0] , Y1_test[pos][1] , 4 ) == latlon2healpix ( lat, lon , 4 ): correct1 += 1

    print("Mean distance : %s km" % np.mean(distances) )
    print("Median distance : %s km" % np.median(distances) )
    print("Confidence interval for mean distance : " , mean_confidence_interval(distances) )
    print("Confidence interval for median distance : " , median_confidence_interval(distances) )
    print("Region accuracy calculated from coordinates (resolution=4): %s" % float(correct1 / float(len(distances))) )
    print("Region accuracy calculated from coordinates (resolution=64): %s" % float(correct2 / float(len(distances))) )
    print("Region accuracy calculated from coordinates (resolution=256): %s" % float(correct3 / float(len(distances))) )
    print("Region accuracy calculated from coordinates (resolution=1024): %s" % float(correct4 / float(len(distances))) )
    print("Accurary under or equal to 161km:", acc_161/test_instances_sz)
    print()
    f.close()

    out_results = open(results_file,"a")
    out_results.write("Mean distance : %s km\n" % np.mean(distances))
    out_results.write("Median distance : %s km\n" % np.median(distances))
    out_results.write("Confidence interval for mean distance : " + str(mean_confidence_interval(distances))+"\n")
    out_results.write("Confidence interval for median distance : " + str(median_confidence_interval(distances))+"\n")
    out_results.write("Region accuracy calculated from coordinates (resolution=4): %s\n" % float(correct1 / float(len(distances))))
    out_results.write("Region accuracy calculated from coordinates (resolution=64): %s\n" % float(correct2 / float(len(distances))))
    out_results.write("Region accuracy calculated from coordinates (resolution=256): %s\n" % float(correct3 / float(len(distances))))
    out_results.write("Region accuracy calculated from coordinates (resolution=1024): %s\n" % float(correct4 / float(len(distances))))
    out_results.write("Accurary under or equal to 161km: "+str(acc_161/test_instances_sz)+"\n\n")
    out_results.close()

    if args.geographicInfo=="yes":

        lc_predicted = predictions[2].argmax(axis=-1)
        lc_real = Y3_test.argmax(axis=-1)

        lc_correct = 0
        for i in range(test_instances_sz):
            if lc_predicted[i] == lc_real[i]:
                lc_correct += 1

        el_predicted = predictions[3].flatten()
        el_real = Y4_test.flatten()
        elevation_mae = mean_absolute_error(el_predicted, el_real)

        ve_predicted = predictions[4].flatten()
        ve_real = Y5_test.flatten()
        vegetation_mae = mean_absolute_error(ve_predicted, ve_real)

        wd_predicted = predictions[5].flatten()
        wd_real = Y6_test.flatten()
        water_dist_mae = mean_absolute_error(wd_predicted, wd_real)

        print("Land Coverage accuracy: ", lc_correct/test_instances_sz)
        print("Elevation - mean absolute eror: ", elevation_mae, " %")
        print("Vegetation - mean absolute eror: ", vegetation_mae, " %")
        print("water distance - mean absolute eror: ", water_dist_mae, " %")

        out_results = open(results_file,"a")
        out_results.write("Land Coverage accuracy: "+str(lc_correct/test_instances_sz)+"\n")
        out_results.write("Elevation - mean absolute eror: "+str(elevation_mae)+" %\n")
        out_results.write("Vegetation - mean absolute eror: "+str(vegetation_mae)+" %\n")
        out_results.write("Water distance - mean absolute eror: "+str(water_dist_mae)+" %\n")
        out_results.close()
