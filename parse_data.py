import numpy
import numpy as np
import os
import struct
from struct import unpack
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import pathlib
import cv2

class Item(object):
    __slots__ = ['x', 'y', 'id', 'cn']

    def __init__(self, x, y, id, cn):
        self.x = x
        self.y = y
        self.id = id
        self.cn = cn


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


def create_dataset(data_path, save_path='./', train_part=0.7, test_part=0.2, validate_part=0.1,
                   max_items_per_class=None, num_classes=None):
    validate_data = []
    test_data = []
    train_data = []
    total_samples = 0
    labels = []
    max_len = 0
    pocket_size = 15000
    for binary_file in os.listdir(data_path):
        class_array = []
        recognize_counter = 0
        class_name = binary_file.split('.bin')[0]
        labels.append(class_name)
        for drawing in unpack_drawings(os.path.join(data_path, binary_file)):
            total_samples += 1
            recognize_counter += drawing['recognized']
            if drawing['recognized']:
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
                # 1. Size normalization.
                lower = np.min(np_ink[:, 0:2], axis=0)
                upper = np.max(np_ink[:, 0:2], axis=0)
                scale = upper - lower
                scale[scale == 0] = 1
                np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
                # 2. Compute deltas.
                # np_ink = np_ink[1:, 0:2] - np_ink[0:-1, 0:2]

                # uncomment for image mirroring by x axes
                # np_ink = [(np_ink[index][0], -np_ink[index][1], np_ink[index][2]) for index in range(len(np_ink))]
                # draw function
                # current_t = 0
                # lines_array = []
                # for stroke in inkarray:
                #     np_ink = np.zeros((len(stroke[0]), 2), dtype=np.float32)
                #     for i in [0, 1]:
                #         np_ink[:len(stroke[0]), i] = stroke[i]
                #     np_ink = [(np_ink[index][0], -np_ink[index][1]) for index in range(len(np_ink))]
                #     lines_array.append(np_ink)
                # # uncomment for image mirroring by x axes
                # # np_ink = [(np_ink[index][0], -np_ink[index][1]) for index in range(len(np_ink))]
                #
                # # show image for debug
                # # test = (((0, 1), (1, 1)), ((1, 2), (2, 1)))
                # # plt.plot(*lines_array)
                # for stroke in lines_array:
                #     plt.plot(*zip(*stroke))
                max_len = max(len(np_ink), max_len)
                class_array.append(Item(np_ink, class_name, drawing['timestamp'], drawing['countrycode']))
                if len(class_array) >= max_items_per_class:
                    break

        print('\nClass name:', class_name)
        print('Items count: ', len(class_array))
        print('Recognized image count:', recognize_counter)
        np.random.shuffle(class_array)
        np.random.shuffle(class_array)
        dataset_size = len(class_array) if max_items_per_class is None else max_items_per_class
        validate_data.extend(class_array[int(dataset_size * (train_part + test_part)):dataset_size])
        test_data.extend(class_array[int(dataset_size * train_part):int(dataset_size * (train_part+test_part))])
        train_data.extend(class_array[:int(dataset_size * train_part)])
        if len(labels) >= num_classes:
            break
    print('Validate', len(validate_data))
    print('Test', len(test_data))
    print('Train', len(train_data))
    print('Total samples: ', total_samples)
    print('Max Len', max_len)

    for data_part in ((train_data, 'train'), (test_data, 'test'), (validate_data, 'validate')):
        pack_data(data_part[0], data_part[1], save_path, pocket_size)

    with open(os.path.join(save_path, 'labels'), 'w') as l_file:
        for label in labels:
            l_file.write(label + '\n')


def pack_data(train_data,  name, save_path, pocket_size):
    pathlib.Path(os.path.join(save_path, name)).mkdir(exist_ok=True)
    for i in range(0, len(train_data), pocket_size):
        with open(os.path.join(save_path, name, '{}_{}.pickle'.format(name, i)), 'wb') as handle:
            if i < len(train_data) - pocket_size:
                pickle.dump(train_data[i:i + pocket_size], handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                pickle.dump(train_data[i:], handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dataset(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class DataProvider:

    def __init__(self, data_path, class_dict, is_conv_nn, max_data_len=None):
        self.data_path = data_path
        self.class_dict = class_dict
        self.max_data_len = max_data_len
        self.is_conv_nn = is_conv_nn
        self.num_classes = len(class_dict.keys())
        self.img_size=100

    def label_vectoring(self, label_text):
        vectorized_label = np.zeros(len(self.class_dict.keys()), dtype=np.int32)
        vectorized_label[self.class_dict[label_text.strip()]] = 1
        return vectorized_label

    def create_x_y(self, data):
        if not self.is_conv_nn:
            x = numpy.zeros((len(data), self.max_data_len, 3), dtype=np.float32)
            y = numpy.zeros((len(data), self.num_classes), dtype=np.int32)
            for index in range(len(data)):
                x_data = data[index].x
                if x_data.shape[1] is 2:
                    x_data = numpy.concatenate((x_data, numpy.zeros((x_data.shape[0], 1))), axis=1)
                x[index] = numpy.concatenate((x_data, numpy.zeros((self.max_data_len - x_data.shape[0], 3))))
                y[index] = self.label_vectoring(data[index].y)
        else:
            x = numpy.zeros((len(data), self.img_size, self.img_size, 1), dtype=np.int32)
            y = numpy.zeros((len(data), self.num_classes), dtype=np.int32)
            for index in range(len(data)):
                x_data = data[index].x
                x_img = np.ones((self.img_size, self.img_size, 1), dtype=np.int32)
                for dot_index in range(x_data.shape[0]):
                    dot = x_data[dot_index]
                    if dot_index > 0 and x_data[dot_index-1][2] == 0:
                        previous_dot = x_data[dot_index-1]
                        cv2.line(x_img,
                                 (int(previous_dot[0]*self.img_size), int(previous_dot[1]*self.img_size)),
                                 (int(dot[0]*self.img_size),          int(dot[1]*self.img_size)),
                                 (0, 0, 0))
                x[index] = x_img
                y[index] = self.label_vectoring(data[index].y)
        return numpy.array(x), numpy.array(y)

    def get_data_batch(self, batch_size=None):
        for data in self.read_data_folder():
            index_list = list(range(0, len(data), batch_size if batch_size is not None else len(data)))
            for i in range(len(index_list)):
                if i < len(index_list)-1:
                    yield self.create_x_y(data[index_list[i]:index_list[i+1]])
                else:
                    yield self.create_x_y(data[index_list[i]:])

    def read_data_folder(self):
        for item in os.listdir(self.data_path):
            yield load_dataset(os.path.join(self.data_path,item))
