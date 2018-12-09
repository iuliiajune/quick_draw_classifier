import os
import time
import tqdm
import numpy
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.python.ops import init_ops

import parse_data as pd


class NetworkBase:

    def __init__(self, label_path, use_gpu, img_size, log_dir=None, pretrained_path=None,
                 optimaizer=tf.train.GradientDescentOptimizer):

        self.pretrained_path = pretrained_path
        self.class_dict = self.read_labels(label_path)
        self.num_classes = len(self.class_dict.keys())
        self.summaries_dir = log_dir
        self.use_gpu = use_gpu
        self.img_size = img_size
        # tf.train.AdamOptimizer
        self.optimaizer = optimaizer
        self.inks, self.targets = self.get_input_tensors()
        self.logits = None

    def get_input_tensors(self):
        with tf.variable_scope('input_x'):
            input_x = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1])
        with tf.variable_scope('input_y'):
            input_y = tf.placeholder(tf.float16, [None, self.num_classes])
        return input_x, input_y

    @staticmethod
    def add_fc_layers(input, size, name):
        with tf.variable_scope(name):
            layer = tf.layers.dense(inputs=input, units=size, activation=tf.nn.relu6, bias_initializer=init_ops.glorot_uniform_initializer())
        return layer

    @staticmethod
    def add_conv_block(input, filter_count, kernel_size, pool_size, pool_strides):
        conv = tf.layers.conv2d(
            inputs=input,
            filters=filter_count,
            kernel_size=[kernel_size, kernel_size],
            padding="same",
            activation=tf.nn.relu6)
        conv = tf.layers.conv2d(
            inputs=conv,
            filters=filter_count,
            kernel_size=[kernel_size, kernel_size],
            padding="same",
            activation=tf.nn.relu6)
        conv = tf.layers.batch_normalization(conv)
        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[pool_size, pool_size], strides=pool_strides)
        return pool

    @staticmethod
    def read_labels(label_path):
        dict = {}
        with open(label_path, 'r') as f:
            class_number = -1
            for item in f.readlines():
                class_name = item.strip()
                class_number += 1
                dict[class_name] = class_number
        return dict

    def train(self, train_data_path, validate_data_path, save_path=None, batch_size=200, learning_rate=0.1, epochs=100,
              print_step=100, validation_step=3):
        with tf.name_scope('total'):
            # The loss function
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.targets,
                                                                                      logits=self.logits))
        tf.summary.scalar('cross_entropy', cross_entropy)
        with tf.name_scope('train'):
            # add an optimiser
            optimiser = self.optimaizer(learning_rate=learning_rate).minimize(cross_entropy)
        saver = tf.train.Saver()
        # load dataset
        train_dp = pd.DataProvider(train_data_path, self.class_dict, self.img_size)
        validate_dp = pd.DataProvider(validate_data_path, self.class_dict, self.img_size)
        # start the session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # with tf.name_scope('validate_accuracy'):
            #     with tf.name_scope('validate_correct_prediction'):
            #         validate_correct_prediction = tf.equal(tf.argmax(self.targets, 1), tf.argmax(self.logits, 1))
            #     with tf.name_scope('validate_accuracy'):
            #         # define an accuracy assessment operation
            #         validate_accuracy = tf.reduce_mean(tf.cast(validate_correct_prediction, tf.float16))
            # tf.summary.scalar('validate_accuracy', validate_accuracy)

            with tf.name_scope('train_accuracy'):
                with tf.name_scope('train_correct_prediction'):
                    correct_prediction = tf.equal(tf.argmax(self.targets, 1), tf.argmax(self.logits, 1))
                with tf.name_scope('train_accuracy'):
                    # define an accuracy assessment operation
                    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float16))
            tf.summary.scalar('train_accuracy', train_accuracy)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(
                self.summaries_dir if self.summaries_dir is not None else './tmp_logdir', sess.graph)

            # initialise the variables
            if self.pretrained_path is None:
                sess.run(tf.global_variables_initializer(),
                         options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
            else:
                saver.restore(sess, self.pretrained_path)

            for epoch in range(1, epochs):
                avg_cost = 0
                prev_arg_cost = None
                step = 0
                start_time = time.time()
                for batch_x, batch_y in tqdm.tqdm(train_dp.get_data_batch(batch_size)):
                    step += 1
                    _, c, summary = sess.run([optimiser, cross_entropy, merged], feed_dict={self.inks: batch_x,
                                                                                            self.targets: batch_y},
                                             options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
                    avg_cost += c
                    train_writer.add_summary(summary, step)
                    if step % print_step == 0:
                        diff = avg_cost-prev_arg_cost if prev_arg_cost is not None else avg_cost
                        prev_arg_cost = avg_cost
                        print(diff/print_step)
                        tf.summary.scalar('mean', diff/print_step)
                print("\nEpoch:", (epoch), "train cost = {:.10f}".format(avg_cost/step))
                if save_path is not None:
                    result_path = saver.save(sess, os.path.join(save_path, "pass_{}.ckpt".format(epoch)))
                    print('Checkpoint saved to', result_path)
                if epoch % validation_step == 0:
                    result = []
                    for valid_x, valid_y in tqdm.tqdm(validate_dp.get_data_batch(batch_size)):
                        validate_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.targets, 1),
                                                                            tf.argmax(self.logits, 1)), tf.float16))
                        res = sess.run(validate_accuracy, feed_dict={self.inks: valid_x, self.targets: valid_y})
                        result.append(res)
                    print('\nValidation', numpy.mean(result))
                print('Epoch time {} s'.format(time.time()-start_time))

    def inference(self, test_data_provider):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # Restore variables from disk.
            saver.restore(sess, self.pretrained_path)
            result = []
            target = []
            for valid_x, valid_y in tqdm.tqdm(test_data_provider.get_data_batch(1)):
                res = sess.run(self.logits, feed_dict={self.inks: valid_x, self.targets: valid_y})
                result.append(numpy.argmax(res))
                target.append(numpy.argmax(valid_y))
        return result, target

    def get_statistics(self, test_data_path):
        test_data_provider = pd.DataProvider(test_data_path, self.class_dict, self.img_size)
        res, target = self.inference(test_data_provider)
        names = [item[0] for item in sorted(self.class_dict.items(), key=lambda kv: kv[1])]
        return classification_report(numpy.array(res), numpy.array(target), target_names=names)
