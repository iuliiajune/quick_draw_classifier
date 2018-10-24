import numpy as np
import os
import struct
from struct import unpack
import matplotlib.pyplot as plt
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
                   max_items_per_class=None):
    validate_data = []
    test_data = []
    train_data = []
    total_samples = 0

    for binary_file in os.listdir(data_path):
        class_array = []
        recognize_counter = 0
        class_name = binary_file.split('.bin')[0]
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
                # uncomment for image mirroring by x axes
                # np_ink = [(np_ink[index][0], -np_ink[index][1], np_ink[index][2]) for index in xrange(len(np_ink))]
                # show image for debug
                # plt.plot(*zip(*np_ink))
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
    print(len(validate_data))
    print(len(test_data))
    print(len(train_data))
    print('Total samples: ', total_samples)

    with open(os.path.join(save_path, 'validate.pickle'), 'wb') as handle:
        pickle.dump(validate_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_path, 'test.pickle'), 'wb') as handle:
        pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_path, 'train.pickle'), 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dataset(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


create_dataset("binary", max_items_per_class=94000)
data = load_dataset('validate.pickle')
