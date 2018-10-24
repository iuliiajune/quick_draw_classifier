import sklearn

import numpy as np
import os
import struct
from struct import unpack
import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'countrycode': countrycode,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break


def parse_data(data_path):
    data = []
    total_samples = 0
    save_path = '/media/viktor/6654680F5467DFF3'
    # create indexes

    for binary_file in os.listdir(data_path):
        class_array = []
        recognize_counter = 0
        for drawing in unpack_drawings(os.path.join(data_path, binary_file)):
            total_samples += 1
            class_name = binary_file.split('.bin')[0]
            recognize_counter += drawing['recognized']
            if drawing['recognized']:
                class_array.append(drawing['key_id'])
            # uncomment for image mirroring by x axes
            # np_ink = [(np_ink[index][0], -np_ink[index][1], np_ink[index][2]) for index in xrange(len(np_ink))]
            # show image for debug
            # plt.plot(*zip(*np_ink))
        data.extend(class_array)
        print 'Class name:', class_name
        print 'Items count: ', len(class_array)
        print 'Recognized image count:', recognize_counter
    print 'Total samples: ', total_samples
    train_part = 0.7
    test_part = 0.2
    validate_part = 0.1

    np.random.shuffle(data)
    dataset_size = len(data)
    train = data[0:int(dataset_size*train_part)]
    test = data[len(train):len(train)+int(dataset_size*test_part)]
    validate = data[len(test)+len(train):]
    print len(train), len(test), len(validate)
    print len(train)+len(test)+len(validate), dataset_size
    for ids_array in ((train, 'train'), (test, 'test'), (validate, 'validate')):
        part_data = []
        for binary_file in os.listdir(data_path):
            for drawing in unpack_drawings(os.path.join(data_path, binary_file)):
                class_name = binary_file.split('.bin')[0]
                id = drawing['key_id']
                if id in ids_array[0]:
                    inkarray = drawing['image']
                    stroke_lengths = [len(stroke[0]) for stroke in inkarray]
                    total_points = sum(stroke_lengths)
                    np_ink = np.zeros((total_points, 3), dtype=np.float32)
                    current_t = 0
                    for stroke in inkarray:
                        for i in [0, 1]:
                            np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
                        current_t += len(stroke[0])
                        np_ink[current_t - 1, 2] = 1  # stroke_end
                    part_data.append((drawing['key_id'], class_name, drawing['countrycode'], np_ink))
        file_name = ids_array[1]+'.pickle'
        with open(os.path.join(save_path, file_name), 'wb') as handle:
            np.pickle.dump(part_data, handle, protocol=np.pickle.HIGHEST_PROTOCOL)


parse_data("binary")
