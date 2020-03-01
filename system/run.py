import os
import sys
import math
import time
import argparse

import numpy as np
import healpy
import rasterio
import pyproj
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense
from keras.layers import LSTM, Bidirectional
from keras.layers.core import Lambda
from keras.layers import concatenate
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from clr_callback import CyclicLR
from batch_generators import *
from bert import bert_embeddings
from elmo import elmo_embeddings
from nn_model import *
from utils import *


sys.setrecursionlimit(3000)

# fix random seed for reproducibility
np.random.seed(0)
os.environ["PYTHONHASHSEED"] = "0"


def set_environment(args):

    if args.wiki == "yes":
        train_data = "../corpora/"+str.upper(args.corpora)+"/"+args.corpora+"-train-wiki.txt"
        test_data = "../corpora/"+str.upper(args.corpora)+"/"+args.corpora+"-test-wiki.txt"
        train_pickle = "../corpora/"+str.upper(args.corpora)+"/"+args.corpora+"-"+args.embeddingType+"-train-wiki.pickle"
        test_pickle = "../corpora/"+str.upper(args.corpora)+"/"+args.corpora+"-"+args.embeddingType+"-test-wiki.pickle"
    else:
        train_data = "../corpora/"+str.upper(args.corpora)+"/"+args.corpora+"-train.txt"
        test_data = "../corpora/"+str.upper(args.corpora)+"/"+args.corpora+"-test.txt"
        train_pickle = "../corpora/"+str.upper(args.corpora)+"/"+args.corpora+"-"+args.embeddingType+"-train.pickle"
        test_pickle = "../corpora/"+str.upper(args.corpora)+"/"+args.corpora+"-"+args.embeddingType+"-test.pickle"

    
    if not os.path.exists("../results/"):
        os.makedirs("../results/")

    if args.wiki == "yes":
        model_name = "../results/"+args.corpora+"-"+args.embeddingType+"-wikipedia.h5"
        distance_file = "../results/"+args.corpora+"-"+args.embeddingType+"-wikipedia.results"
        results_file = "../results/"+args.corpora+"-"+args.embeddingType+"-wikipedia.txt"

    elif args.geographicInfo == "yes":
        model_name = "../results/"+args.corpora+"-"+args.embeddingType+"-geoproperties.h5"
        distance_file = "../results/"+args.corpora+"-"+args.embeddingType+"-geoproperties.results"
        results_file = "../results/"+args.corpora+"-"+args.embeddingType+"-geoproperties.txt"

    else:
        model_name = "../results/"+args.corpora+"-"+args.embeddingType+".h5"
        distance_file = "../results/"+args.corpora+"-"+args.embeddingType+".results"
        results_file = "../results/"+args.corpora+"-"+args.embeddingType+".txt"

    return train_data, test_data, train_pickle, test_pickle, model_name, distance_file, results_file


def get_values( lc , el , ve , lc_d , el_d , ve_d , wd, wd_d, wd_max, lat, lon ):
    src = pyproj.Proj(lc.crs)
    lonlat = pyproj.Proj(init="epsg:4326")
    east,north = pyproj.transform(lonlat, src, lon, lat)
    row, col = lc.index(north, east)
    lc_v = lc_d[row,col]

    sric = pyproj.Proj(el.crs)
    lonlat = pyproj.Proj(init="epsg:4326")
    east,north = pyproj.transform(lonlat, src, lon, lat)
    row, col = el.index(north, east)
    el_v = el_d[row,col]

    src = pyproj.Proj(ve.crs)
    lonlat = pyproj.Proj(init="epsg:4326")
    east,north = pyproj.transform(lonlat, src, lon, lat)
    row, col = ve.index(north, east)
    ve_v = ve_d[row,col]

    src = pyproj.Proj(wd.crs)
    lonlat = pyproj.Proj(init="epsg:4326")
    east,north = pyproj.transform(lonlat, src, lon, lat)
    row, col = wd.index(north, east)
    wd_v = wd_d[row,col]

    if ve_v > 100.0 : ve_v = 0.0
    else : ve_v = ve_v / 100.0
    if ( el_v == -9999 ) or ( el_v == 9998 ) : el_v = 0.0
    else: el_v = el_v / 8850.0
    wd_v = wd_v / wd_max
    return [ keras.utils.to_categorical(lc_v-1, num_classes=20) ,
             np.array([ el_v, ve_v, wd_v]) ]




if __name__ == "__main__":

    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddingType", type=str, default="elmo")
    parser.add_argument("--geographicInfo", type=str, default="no")
    parser.add_argument("--corpora", type=str, default="wotr")
    parser.add_argument("--wiki", type=str, default="no")
    args = parser.parse_args()

    print("Starting the toponym resolution system. Considering the " + str.upper(args.corpora) + " corpus.")

    train_data, test_data, output_file1, output_file2, model_name, distance_file, results_file = set_environment(args)

    mention_size = 5
    sentence_size = 50
    text_size = 500
    resolution = math.pow(4,4)
    batch_size = 32
    epochs = 200

    print(str.upper(args.embeddingType)+" embeddings: Sarting computation...")
 
    if args.embeddingType == "elmo":
        embedding_size = 1024
        Y1_train = elmo_embeddings(args, train_data, output_file1, batch_size, text_size,
                    mention_size, sentence_size)
        Y1_test = elmo_embeddings(args, test_data, output_file2, batch_size, text_size,
                    mention_size, sentence_size)
 
    elif args.embeddingType == "bert":
        embedding_size = 768
        Y1_train = bert_embeddings(args, train_data, output_file1, batch_size, text_size,
                    mention_size, sentence_size, embedding_size)
        Y1_test = bert_embeddings(args, test_data, output_file2, batch_size, text_size,
                    mention_size, sentence_size, embedding_size)

    print(str.upper(args.embeddingType)+" embedding: Computation finished.")

    train_instances_sz = len(Y1_train)
    test_instances_sz = len(Y1_test)
    print("Train instances:", train_instances_sz)
    print("Test instances:", test_instances_sz)

  
    if args.geographicInfo=="yes":
        print("Reading land coverage data...")
        if args.corpora=="wotr":
            land_coverage_file = rasterio.open("../geoproperties-files/historic_landcover_hd_1850.tif")
        elif args.corpora=="lgl" or args.corpora=="spatialML":
            land_coverage_file = rasterio.open("../geoproperties-files/gm_lc_v3.tif")
        land_coverage_data = land_coverage_file.read(1)
        print("Reading elevation data...")
        elevation_file= rasterio.open("../geoproperties-files/gm_el_v1.tif")
        elevation_data = elevation_file.read(1)
        print("Reading vegetation coverage data...")
        vegetation_file = rasterio.open("../geoproperties-files/gm_ve_v2.tif")
        vegetation_data = vegetation_file.read(1)
        print("Reading water distance data...")
        if args.corpora=="wotr":
            water_dist_file = rasterio.open("../geoproperties-files/distance-historic_landcover_hd_1850.tif")
        elif args.corpora=="lgl" or args.corpora=="spatialML":
            water_dist_file = rasterio.open("../geoproperties-files/distance-gm_lc_v3.tif")
        water_dist_data = water_dist_file.read(1)
        wd_max = np.amax(water_dist_data)
        print("Geo-Information reading completed")

        lc_data = []
        el_data = []
        ve_data = []
        wd_data = []

    encoder = LabelEncoder()
    region_codes = []
    for coordinates in (Y1_train+Y1_test):
        region = latlon2healpix(coordinates[0], coordinates[1], resolution)
        region_codes.append(region)

        if args.geographicInfo=="yes":
            [lc, [el, ve, wd]] = get_values(land_coverage_file, elevation_file, vegetation_file , land_coverage_data, elevation_data, vegetation_data, water_dist_file, water_dist_data, wd_max, coordinates[0], coordinates[1])
            lc_data.append(lc)
            el_data.append([el])
            ve_data.append([ve])
            wd_data.append([wd])

    classes = to_categorical(encoder.fit_transform(region_codes))
  
    Y1_train = np.array(Y1_train)
    Y1_test = np.array(Y1_test)
    print("Y1_train shape:",Y1_train.shape)
    print("Y1_test shape:",Y1_test.shape)

    Y2_train = classes[:train_instances_sz]
    Y2_test = classes[-test_instances_sz:]
    print("Y2_train shape:",Y2_train.shape)
    print("Y2_test shape:",Y2_test.shape)

    if args.geographicInfo=="yes":
  
        Y3_train = lc_data[:train_instances_sz]
        Y3_test = lc_data[-test_instances_sz:]
        Y3_train = np.array(Y3_train)
        Y3_test = np.array(Y3_test)
        print("Y3_train shape:",Y3_train.shape)
        print("Y3_test shape:",Y3_test.shape)

        Y4_train = el_data[:train_instances_sz]
        Y4_test = el_data[-test_instances_sz:]
        Y4_train = np.array(Y4_train)
        Y4_test = np.array(Y4_test)
        print("Y4_train shape:",Y4_train.shape)
        print("Y4_test shape:",Y4_test.shape)

        Y5_train = ve_data[:train_instances_sz]
        Y5_test = ve_data[-test_instances_sz:]
        Y5_train = np.array(Y5_train)
        Y5_test = np.array(Y5_test)
        print("Y5_train shape:",Y5_train.shape)
        print("Y5_test shape:",Y5_test.shape)

        Y6_train = wd_data[:train_instances_sz]
        Y6_test = wd_data[-test_instances_sz:]
        Y6_train = np.array(Y6_train)
        Y6_test = np.array(Y6_test)
        print("Y6_train shape:",Y6_train.shape)
        print("Y6_test shape:",Y6_test.shape)
  
    print("Build model...")
  
    region_list = [i for i in range(Y2_train.shape[1])]
    region_classes = encoder.inverse_transform(region_list)
    codes_matrix = []
    for i in range(len(region_classes)):
        [xs, ys, zs] = healpy.pix2vec( int(resolution), region_classes[i] )
        codes_matrix.append([xs, ys, zs])
  
    if args.geographicInfo=="yes":
  
        land_cover_matrix = []
        elevation_matrix = []
        vegetation_matrix = []
        water_dist_matrix = []
  
        for i in range(len(codes_matrix)):
            [xs, ys, zs] = codes_matrix[i]
            lat = float( math.atan2(zs, math.sqrt(xs * xs + ys * ys)) * 180.0 / math.pi )
            lon = float( math.atan2(ys, xs) * 180.0 / math.pi )
            [lc, [el, ve, wd]] = get_values(land_coverage_file, elevation_file, vegetation_file , land_coverage_data, elevation_data, vegetation_data, water_dist_file, water_dist_data, wd_max, lat, lon)
            land_cover_matrix.append(lc.tolist())
            elevation_matrix.append([el])
            vegetation_matrix.append([ve])
            water_dist_matrix.append([wd])
            
        codes_matrix = K.variable(value=codes_matrix, dtype="float32")
        land_cover_matrix = K.variable(value=land_cover_matrix, dtype="float32")
        elevation_matrix = K.variable(value=elevation_matrix, dtype="float32")
        vegetation_matrix = K.variable(value=vegetation_matrix, dtype="float32")
        water_dist_matrix = K.variable(value=water_dist_matrix, dtype="float32")

        input_mention = Input(shape=(mention_size, embedding_size), dtype="float32", name="mention")
        input_sentence = Input(shape=(sentence_size, embedding_size), dtype="float32", name="sentence")
        input_text = Input(shape=(text_size, embedding_size), dtype="float32", name="text")
  
        x1 = Bidirectional(LSTM(512, activation="pentanh", recurrent_activation="pentanh"))(input_mention)
        x2 = Bidirectional(LSTM(512, activation="pentanh", recurrent_activation="pentanh"))(input_sentence)
        x3 = Bidirectional(LSTM(512, activation="pentanh", recurrent_activation="pentanh"))(input_text)
        concat1 = concatenate([x1, x2, x3], axis=1)
        x = Dense(512, activation="pentanh")(concat1)
        auxiliary_output = Dense(Y2_train.shape[1], activation="softmax", name="aux_output")(x)
        cube_prob = Lambda(cube)(auxiliary_output)
        main_output = Lambda(lambda cube_prob: K.dot(cube_prob, codes_matrix), name="main_output")(cube_prob)
        lc_output = Lambda(lambda cube_prob: K.dot(cube_prob, land_cover_matrix), name="lc_output")(cube_prob)
        el_output = Lambda(lambda cube_prob: K.dot(cube_prob, elevation_matrix), name="el_output")(cube_prob)
        ve_output = Lambda(lambda cube_prob: K.dot(cube_prob, vegetation_matrix), name="ve_output")(cube_prob)
        wd_output = Lambda(lambda cube_prob: K.dot(cube_prob, water_dist_matrix), name="wd_output")(cube_prob)

        model = Model(inputs=[input_mention, input_sentence, input_text],
            outputs=[main_output, auxiliary_output, lc_output, el_output, ve_output, wd_output])
        model.summary()
  
        print("Training the model...")
        early_stop = EarlyStopping(monitor="loss", patience=5, verbose=1)
        clr = CyclicLR(base_lr=0.00001, max_lr=0.0001, mode="triangular", step_size=(2.0 - 8.0) * (len(Y1_train)/epochs))
        adamOpt = keras.optimizers.Adam(lr=0.0001, amsgrad=True, clipnorm=1.)
        model.compile(optimizer=adamOpt, loss={"main_output": geodistance_tensorflow, "aux_output": "categorical_crossentropy", "lc_output": "categorical_crossentropy", "el_output":"mean_absolute_error", "ve_output":"mean_absolute_error", "wd_output":"mean_absolute_error"},
            loss_weights=[0.50, 50.0, 0.25, 0.25, 0.25, 0.25],
            metrics={"main_output": geodistance_tensorflow, "aux_output": "categorical_accuracy", "lc_output": "categorical_accuracy", "el_output":"mean_absolute_error", "ve_output":"mean_absolute_error", "wd_output":"mean_absolute_error"})
  
        model.fit_generator(
                batch_generator_geoproperties(output_file1, train_instances_sz, Y1_train, Y2_train, Y3_train, Y4_train, Y5_train, Y6_train, batch_size),
                steps_per_epoch=train_instances_sz/batch_size, epochs=epochs, callbacks=[clr, early_stop],
                validation_steps=test_instances_sz/batch_size,
                validation_data=batch_generator_geoproperties(output_file2, test_instances_sz, Y1_test, Y2_test, Y3_test, Y4_test, Y5_test, Y6_test, batch_size))
  
        model.save(model_name)
        # model = load_model(model_name, custom_objects={"geodistance_tensorflow":geodistance_tensorflow, "codes_matrix":codes_matrix, "land_cover_matrix":land_cover_matrix, "elevation_matrix":elevation_matrix, "vegetation_matrix":vegetation_matrix, "water_dist_matrix":water_dist_matrix})
        print("Computing predictions...")
        predictions = model.predict_generator(batch_generator_geoproperties(output_file2, test_instances_sz, Y1_test, Y2_test, 
                                            Y3_test, Y4_test, Y5_test, Y6_test, batch_size), steps=test_instances_sz/batch_size)
  
        print("Computing accuracy...\n")
        get_results(args, test_data, distance_file, results_file, predictions, test_instances_sz, 
                    encoder, Y1_test, Y2_test, Y3_test, Y4_test, Y5_test, Y6_test)
  
    if args.geographicInfo=="no":

        if args.wiki == "yes":
            sample_weights_vec = get_sample_weights(train_data)
            sample_weights_vec = np.asarray(sample_weights_vec)
  
        codes_matrix = K.variable(value=codes_matrix, dtype="float32")
  
        input_mention = Input(shape=(mention_size, embedding_size), dtype="float32", name="mention")
        input_sentence = Input(shape=(sentence_size, embedding_size), dtype="float32", name="sentence")
        input_text = Input(shape=(text_size, embedding_size), dtype="float32", name="text")
  
        x1 = Bidirectional(LSTM(512, activation="pentanh", recurrent_activation="pentanh"))(input_mention)
        x2 = Bidirectional(LSTM(512, activation="pentanh", recurrent_activation="pentanh"))(input_sentence)
        x3 = Bidirectional(LSTM(512, activation="pentanh", recurrent_activation="pentanh"))(input_text)
        concat1 = concatenate([x1, x2, x3], axis=1)
        x = Dense(512, activation="pentanh")(concat1)
        auxiliary_output = Dense(Y2_train.shape[1], activation="softmax", name="aux_output")(x)
        cube_prob = Lambda(cube)(auxiliary_output)
        main_output = Lambda(lambda cube_prob: K.dot(cube_prob, codes_matrix), name="main_output")(cube_prob)
  
        model = Model(inputs=[input_mention, input_sentence, input_text],
            outputs=[main_output, auxiliary_output])
        model.summary()
 
        print("Training the model...")
        early_stop = EarlyStopping(monitor="loss", patience=5, verbose=1)
        clr = CyclicLR(base_lr=0.00001, max_lr=0.0001, mode="triangular", step_size=(2.0 - 8.0) * (len(Y1_train)/epochs))
        adamOpt = keras.optimizers.Adam(lr=0.0001, amsgrad=True, clipnorm=1.)
        model.compile(optimizer=adamOpt, loss={"main_output": geodistance_tensorflow, "aux_output": "categorical_crossentropy"},
            loss_weights=[0.50, 50.0],
            metrics={"main_output": geodistance_tensorflow, "aux_output": "categorical_accuracy"})
  
        if args.wiki == "yes":
            model.fit_generator(
                    batch_generator_wikipedia(output_file1, train_instances_sz, Y1_train, Y2_train, 1, batch_size, sample_weights_vec),
                    steps_per_epoch=train_instances_sz/batch_size, epochs=epochs, callbacks=[clr, early_stop],
                    validation_steps=test_instances_sz/batch_size,
                    validation_data=batch_generator_wikipedia(output_file2, test_instances_sz, Y1_test, Y2_test, 0, batch_size, None))
 
        elif args.wiki == "no":
            model.fit_generator(
                    batch_generator(output_file1, train_instances_sz, Y1_train, Y2_train, batch_size),
                    steps_per_epoch=train_instances_sz/batch_size, epochs=epochs, callbacks=[clr, early_stop],
                    validation_steps=test_instances_sz/batch_size,
                    validation_data=batch_generator(output_file2, test_instances_sz, Y1_test, Y2_test, batch_size))
  
        model.save(model_name)
        # model = load_model(model_name, custom_objects={"geodistance_tensorflow": geodistance_tensorflow, "codes_matrix": codes_matrix})
        print("Computing predictions...")
  
        if args.wiki == "yes":
            predictions = model.predict_generator(batch_generator_wikipedia(output_file2, test_instances_sz, Y1_test, Y2_test, 0, batch_size, None), steps=test_instances_sz/batch_size)
        elif args.wiki == "no":
            predictions = model.predict_generator(batch_generator(output_file2, test_instances_sz, Y1_test, Y2_test, batch_size), steps=test_instances_sz/batch_size)
  
        print("Computing accuracy...\n")
        get_results(args, test_data, distance_file, results_file, predictions, test_instances_sz, 
                    encoder, Y1_test, Y2_test, Y3_test=None, Y4_test=None, Y5_test=None, Y6_test=None)

    end = time.time()
    print("Execution time:",round((end-start)/60,5)/60,"hours.")
    out_results = open(results_file,"a")
    out_results.write("Execution time:"+str(round((end-start)/60,5)/60)+"hours.")
    out_results.close()