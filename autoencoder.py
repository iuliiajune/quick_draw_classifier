import os
import pathlib
import time

import numpy
import tensorflow as tf
import tqdm

import parse_data as pd
from NetworkBase import NetworkBase

class AutoencoderNetwork(NetworkBase):

    def __init__(self, label_path, use_gpu, img_size, sizes, log_dir=None, pretrained_path=None,
                 optimaizer=tf.train.GradientDescentOptimizer):
        self.sizes = sizes
        self.blocks = []
        super().__init__(label_path, use_gpu, img_size, log_dir, pretrained_path,
                 optimaizer)

    @staticmethod
    def add_encode_block(input, filter_count, kernel_size, pool_size, pool_strides, block_number):
        tensors = list()
        tensors.append(input)

        noise_tensor = tf.placeholder_with_default(tf.zeros_like(input, dtype=tf.float32),
                                                   shape=input.shape)
        conv = tf.clip_by_value(tf.add(noise_tensor, input),
                                tf.reduce_min(input),
                                tf.reduce_max(input))
        tensors.append(noise_tensor)

        conv = tf.layers.conv2d(
            inputs=conv,
            filters=filter_count,
            kernel_size=[kernel_size, kernel_size],
            padding="same",
            activation=tf.nn.relu6)
        tensors.append(conv)

        conv = tf.layers.conv2d(
            inputs=conv,
            filters=filter_count,
            kernel_size=[kernel_size, kernel_size],
            padding="same",
            activation=tf.nn.relu6)
        tensors.append(conv)

        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[pool_size, pool_size], strides=pool_strides)
        tensors.append(pool)
        return tensors, pool

    @staticmethod
    def add_decode_block(filter_count, kernel_size, pool_size, output_chanels_count, tensors, block_number):
        input_tensor = tensors[-1]
        (_, input_shape_x, input_shape_y, _) = input_tensor.get_shape()
        unconv = tf.image.resize_images(input_tensor,
                                        size=(input_shape_x*pool_size, input_shape_y*pool_size),
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        tensors.append(unconv)

        unconv = tf.layers.conv2d_transpose(unconv,
                                            filters=filter_count,
                                            kernel_size=[kernel_size, kernel_size],
                                            padding='SAME',
                                            activation=tf.nn.relu6)
        tensors.append(unconv)

        unconv = tf.layers.conv2d_transpose(unconv,
                                            filters=output_chanels_count,
                                            kernel_size=[kernel_size, kernel_size],
                                            padding='SAME',
                                            activation=tf.nn.relu6)
        tensors.append(unconv)
        return tensors

    def train_encoder_block(self, train_dp, input_tensor, output_tensor,  tmp_path, block_number,
                            epochs, batch_size, print_step, learning_rate, sess, block_tensors, loss):
        tmp_ckpt_path = os.path.join(tmp_path, 'block_{}'.format(block_number))
        pathlib.Path(tmp_ckpt_path).mkdir(exist_ok=True)
        log_dir = os.path.join(tmp_ckpt_path, 'log_dir')

        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_tensor, logits=output_tensor))
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=input_tensor, logits=output_tensor))

        trainable_vars = []
        for tensor in block_tensors[2:]:
            tensor_scope = os.path.split(tensor.name)[0]
            trainable_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tensor_scope)
            trainable_vars.extend(trainable_collection)
        optimiser = self.optimaizer(learning_rate=learning_rate).minimize(loss, var_list=trainable_vars)
        sess.run(tf.global_variables_initializer(),
                 options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
        for epoch in range(epochs):
            avg_cost = 0
            prev_arg_cost = None
            step = 0
            start_time = time.time()
            for batch_x, batch_y in tqdm.tqdm(train_dp.get_data_batch(batch_size)):
                noise_factor = 0.5
                noise_tensor = block_tensors[1]
                noise_shape = [len(batch_x)]+noise_tensor._shape_as_list()[1:]
                noise = noise_factor*numpy.random.randn(*noise_shape)
                step += 1
                _, c = sess.run([optimiser, loss], feed_dict={self.inks: batch_x,
                                                              noise_tensor: noise},
                                options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
                avg_cost += c
                if step % print_step == 0:
                    diff = avg_cost-prev_arg_cost if prev_arg_cost is not None else avg_cost
                    prev_arg_cost = avg_cost
                    print(diff/print_step)
                    # tf.summary.scalar('mean', diff/print_step)
            print("\nEpoch:", (epoch), "train cost = {:.10f}".format(avg_cost/step))
            print('Epoch time {} s'.format(time.time() - start_time))
            # block_dataset = self.create_block_dataset(sess, train_dp, input_tensor, output_tensor, tmp_ckpt_path)
        # return result_path, block_dataset

    def create_block_dataset(self, sess, train_dp, input_tensor, output_tensor, save_path):
        class_array = []
        for batch_x, batch_y in tqdm.tqdm(train_dp.get_data_batch(1)):
            infer_res = sess.run(output_tensor, feed_dict={input_tensor: batch_x})
            class_array.append(pd.Item(infer_res, train_dp.get_class_name(batch_y), "", ""))

        new_dataset_path = os.path.join(save_path, 'train')
        pd.pack_data(class_array, 'train', save_path, 15000)
        return new_dataset_path

    def train(self, train_data_path, validate_data_path, save_path=None, batch_size=200, learning_rate=0.1, epochs=100,
              print_step=100, validation_step=10):
        tmp_path = os.path.join(save_path, 'tmp')
        pathlib.Path(save_path).mkdir(exist_ok=True)
        pathlib.Path(tmp_path).mkdir(exist_ok=True)
        for d in ['/cpu:0'] if not self.use_gpu else ["/device:GPU:0"]:
            with tf.device(d):
                input_tensor = self.inks
                blocks = []
                losses = []
                for block_number in range(len(self.sizes)):
                    (filter_count, kernel_size, pool_size, pool_strides) = self.sizes[block_number]
                    (_, _, _, output_channels_count) = input_tensor.get_shape()
                    block_tensors, encoder = self.add_encode_block(input_tensor,
                                                           filter_count,
                                                           kernel_size,
                                                           pool_size,
                                                           pool_strides,
                                                           block_number)
                    block_tensors = self.add_decode_block(filter_count,
                                                           kernel_size,
                                                           pool_size,
                                                           output_channels_count,
                                                           block_tensors,
                                                           block_number)
                    output_tensor = block_tensors[-1]

                    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=input_tensor,
                                                                                  logits=output_tensor))
                    input_tensor = encoder
                    losses.append(loss)
                    blocks.append(block_tensors)

                self.logits = self.add_fc_layers(input=tf.layers.flatten(input_tensor), size=self.num_classes, name="hidden_fc_last")
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.targets,
                                                                                          logits=self.logits))

                train_dp = pd.DataProvider(train_data_path, self.class_dict, self.img_size)
                validate_dp = pd.DataProvider(validate_data_path, self.class_dict, self.img_size)

                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                    train_writer = tf.summary.FileWriter(self.summaries_dir, sess.graph)
                    saver = tf.train.Saver()

                    #train encoder
                    for block_number in range(len(blocks)):
                        block = blocks[block_number]
                        input_tensor, output_tensor, = block[0], block[-1]
                        self.train_encoder_block(train_dp, input_tensor, output_tensor, tmp_path, block_number, epochs,
                                                 batch_size, print_step, learning_rate, sess, block, losses[block_number])
                    #train fc
                    tensor_scope = os.path.split(self.logits.name)[0]
                    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tensor_scope)

                    optimiser = self.optimaizer(learning_rate=learning_rate).minimize(cross_entropy, var_list=trainable_vars)
                    for epoch in range(epochs):
                        avg_cost = 0
                        prev_arg_cost = None
                        step = 0
                        start_time = time.time()
                        for batch_x, batch_y in tqdm.tqdm(train_dp.get_data_batch(batch_size)):
                            step += 1
                            _, c = sess.run([optimiser, cross_entropy], feed_dict={self.inks: batch_x,
                                                                                   self.targets: batch_y},
                                                     options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
                            avg_cost += c
                            # train_writer.add_summary(summary, step)
                            if step % print_step == 0:
                                diff = avg_cost-prev_arg_cost if prev_arg_cost is not None else avg_cost
                                prev_arg_cost = avg_cost
                                print(diff/print_step)
                                # tf.summary.scalar('mean', diff/print_step)
                        print("\nEpoch:", (epoch), "train cost = {:.10f}".format(avg_cost/step))
                        if save_path is not None:
                            result_path = saver.save(sess, os.path.join(save_path, "pass_{}.ckpt".format(epoch)))
                            print('Checkpoint saved to', result_path)
                        if epoch % validation_step == 0:
                            result = []
                            for valid_x, valid_y in tqdm.tqdm(validate_dp.get_data_batch(batch_size)):
                                validate_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.targets, 1),
                                                                                    tf.argmax(self.logits, 1)), tf.float16))
                                res = sess.run(validate_accuracy, feed_dict={self.inks: valid_x,
                                                                             self.targets: valid_y})
                                result.append(res)
                            print('\nValidation', numpy.mean(result))
                        print('Epoch time {} s'.format(time.time()-start_time))
                    result_path = saver.save(sess, os.path.join(save_path, "pass.ckpt"))
                    print('Checkpoint saved to', result_path)
