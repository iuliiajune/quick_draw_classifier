import numpy
import numpy as np
import os
import struct
from struct import unpack
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle


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
                class_array.append((drawing['timestamp'], class_name, drawing['countrycode'], np_ink))

                # 1. Size normalization.
                lower = np.min(np_ink[:, 0:2], axis=0)
                upper = np.max(np_ink[:, 0:2], axis=0)
                scale = upper - lower
                scale[scale == 0] = 1
                np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
                # 2. Compute deltas.
                np_ink = np_ink[1:, 0:2] - np_ink[0:-1, 0:2]

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
                class_array.append((drawing['timestamp'], class_name, drawing['countrycode'], np_ink))
                if len(class_array) >= max_items_per_class:
                    break

        print('\nClass name:', class_name)
        print('Items count: ', len(class_array))
        print('Recognized image count:', recognize_counter)
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
    with open(os.path.join(save_path, 'validate.pickle'), 'wb') as handle:
        pickle.dump(validate_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_path, 'test.pickle'), 'wb') as handle:
        pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_path, 'train.pickle'), 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_path, 'labels'), 'w') as l_file:
        for label in labels:
            l_file.write(label + '\n')


def load_dataset(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class DataProvider:

    def __init__(self, data_path, class_dict, max_data_len):
        self.data = load_dataset(data_path)
        self.class_dict = class_dict
        self.data_size = len(self.data)
        self.max_data_len = max_data_len
        self.num_classes = len(class_dict.keys())

    def label_vectoring(self, label_text):
        vectorized_label = np.zeros(len(self.class_dict.keys()), dtype=np.int32)
        vectorized_label[self.class_dict[label_text.strip()]] = 1
        return vectorized_label

    def create_x_y(self, data):
        x = numpy.zeros((len(data), self.max_data_len, 3), dtype=np.float32)
        y = numpy.zeros((len(data), self.num_classes), dtype=np.int32)
        for index in range(len(data)):
            x_data = data[index][3]
            if x_data.shape[1] is 2:
                x_data = numpy.concatenate((x_data, numpy.zeros((x_data.shape[0], 1))), axis=1)
            x[index] = numpy.concatenate((x_data, numpy.zeros((self.max_data_len - x_data.shape[0], 3))))
            y[index] = self.label_vectoring(data[index][1])
        return numpy.array(x), numpy.array(y)

    def get_data_batch(self, batch_size=None):
        index_list = list(range(0, len(self.data), batch_size if batch_size is not None else len(self.data)))
        for i in range(len(index_list)):
            if i < len(index_list)-1:
                yield self.create_x_y(self.data[index_list[i]:index_list[i+1]])
            else:
                yield self.create_x_y(self.data[index_list[i]:])
