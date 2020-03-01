import pickle
import numpy as np


def get_sample_weights(filename):
    file = open(filename, "r")
    sample_weights_vec = []
    for line in file:
        content = line.split("\t")
        id_source = int(content[0])

        if id_source == 0:
            sample_weights_vec.append(1.0)
        elif id_source == 1:
            sample_weights_vec.append(0.5)

    return sample_weights_vec


def batch_generator(file_name, size, Y1, Y2, batch_size):
    while 1:
        file=open(file_name, 'rb')
        for i in range(0, size, batch_size):
            aux = pickle.load(file)
            X1 = np.asarray(aux[0])
            X2 = np.asarray(aux[1])
            X3 = np.asarray(aux[2])
            yield [X1, X2, X3],[Y1[i:i+batch_size], Y2[i:i+batch_size]]
        file.close()


def batch_generator_geoproperties(file_name, size, Y1, Y2, Y3, Y4, Y5, Y6, batch_size):
    while 1:
        file=open(file_name, 'rb')
        for i in range(0,size,batch_size):
            aux = pickle.load(file)
            X1 = np.asarray(aux[0])
            X2 = np.asarray(aux[1])
            X3 = np.asarray(aux[2])
            yield [X1, X2, X3],[Y1[i:i+batch_size], Y2[i:i+batch_size],
                                Y3[i:i+batch_size], Y4[i:i+batch_size],
                                Y5[i:i+batch_size], Y6[i:i+batch_size]]
        file.close()


def batch_generator_wikipedia(file_name, size, Y1, Y2, weight_flag,
                         batch_size, sample_weights_vec=None):

        while 1:
            file=open(file_name, 'rb')
            for i in range(0,size,batch_size):
                aux = pickle.load(file)
                X1 = np.asarray(aux[0])
                X2 = np.asarray(aux[1])
                X3 = np.asarray(aux[2])
                if weight_flag == 0:
                    yield [X1, X2, X3], [Y1[i:i+batch_size], Y2[i:i+batch_size]]
                elif weight_flag == 1:
                    yield [X1, X2, X3], [Y1[i:i+batch_size], Y2[i:i+batch_size]], [sample_weights_vec[i:i+batch_size], sample_weights_vec[i:i+batch_size]]
            file.close()
