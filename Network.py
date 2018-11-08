import os
import numpy
import tensorflow as tf
import parse_data as pd
import tqdm
from sklearn.metrics import classification_report


class Network:
    def _get_input_tensors_fc(self):
        with tf.variable_scope('input_x'):
            input_x = tf.placeholder(tf.float32, [None, self.max_data_len, 3])
        with tf.variable_scope('input_y'):
            input_y = tf.placeholder(tf.int32, [None, self.num_classes])
        return input_x, input_y

    def _get_input_tensors_conv(self):
        with tf.variable_scope('input_x'):
            input_x = tf.placeholder(tf.float32, [None, self.img_size, self.img_size])
        with tf.variable_scope('input_y'):
            input_y = tf.placeholder(tf.int32, [None, self.num_classes])
        return input_x, input_y

    def _add_fc_layers(self, inks, size, name):
        with tf.variable_scope(name):
            return tf.contrib.layers.fully_connected(inks, size)

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

    def __init__(self, shapes, label_path, use_gpu, log_dir, pretrained_path, max_data_len, is_conv_nn=True):
        self.max_data_len = max_data_len
        self.pretrained_path = pretrained_path
        self.class_dict = self.read_labels(label_path)
        self.num_classes = len(self.class_dict.keys())
        shapes.append(self.num_classes)
        self.shapes = shapes
        self.summaries_dir = log_dir
        self.use_gpu = use_gpu
        self.is_conv_nn = is_conv_nn
        self.img_size=100

        if not self.is_conv_nn:
            self.create_fc_nn()
        else:
            self.create_cnn_nn()

    def create_fc_nn(self):
        for d in ['/cpu:0'] if not self.use_gpu else ["/device:GPU:0"]:
            with tf.device(d):
                self.inks, self.targets = self._get_input_tensors_fc()
                flatten = tf.layers.flatten(self.inks, name='flatten_data')
                logits = self._add_fc_layers(flatten, self.shapes[0], 'hidden_fc_0')
                for shape_index in range(1, len(self.shapes)):
                    logits = self._add_fc_layers(logits, self.shapes[shape_index],
                                                 'hidden_fc_{}'.format(shape_index))
                self.logits = logits

    def create_cnn_nn(self):
        for d in ['/cpu:0'] if not self.use_gpu else ["/device:GPU:0"]:
            with tf.device(d):
                logits

                
                self.logits = self._add_fc_layers(logits, self.shapes[-1],
                                                 'hidden_fc_{}'.format(-1))

    def train(self, train_data_path, validate_data_path, save_path=None, batch_size=200, learning_rate=0.1,
              epochs=100):
        with tf.name_scope('total'):
            # The loss function
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.targets,
                                                                                      logits=self.logits))
        tf.summary.scalar('cross_entropy', cross_entropy)
        with tf.name_scope('train'):
            # add an optimiser
            optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # load dataset
        train_dp = pd.DataProvider(train_data_path, self.class_dict, self.is_conv_nn, self.max_data_len)
        # start the session
        with tf.Session() as sess:

            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    correct_prediction = tf.equal(tf.argmax(self.targets, 1), tf.argmax(self.logits, 1))
                with tf.name_scope('accuracy'):
                    # define an accuracy assessment operation
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.summaries_dir, sess.graph)

            # initialise the variables
            if self.pretrained_path is None:
                sess.run(tf.global_variables_initializer(),
                         options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
            else:
                saver.restore(sess, self.pretrained_path)
            print_step = 20
            for epoch in range(epochs):
                avg_cost = 0
                prev_arg_cost = None
                step = 0
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
                print("Epoch:", (epoch + 1), "train cost =", "{:.10f}".format(avg_cost/step))
                if save_path is not None:
                    result_path = saver.save(sess, os.path.join(save_path, "pass_{}.ckpt".format(epoch)))
                    print('Checkpoint saved to ', result_path)

            validate_dp = pd.DataProvider(validate_data_path, self.class_dict, self.is_conv_nn, self.max_data_len)
            result = []
            for valid_x, valid_y in tqdm.tqdm(validate_dp.get_data_batch()):
                res = sess.run(accuracy, feed_dict={self.inks: valid_x, self.targets: valid_y})
                result.append(res)
                print(res)
            # print(numpy.mean(result))

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
        test_data_provider = pd.DataProvider(test_data_path, self.class_dict, self.is_conv_nn, self.max_data_len)
        res, target = self.inference(test_data_provider)
        names = [item[0] for item in sorted(self.class_dict.items(), key=lambda kv: kv[1])]
        return classification_report(numpy.array(res), numpy.array(target), target_names=names)
