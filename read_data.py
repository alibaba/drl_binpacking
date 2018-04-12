# -*- coding: UTF-8 -*-
import numpy as np
from collections import namedtuple
import threading
import tensorflow as tf
import utils

# Define a namedtuple named "binpacking" to store sample data
binpacking = namedtuple('binpacking', ['o', 'x', 'b', 'name'])


def read_dataset(path, max_length=10):
    """
    Read data from file
    :param path: The path of data file
    :param max_length: The maximum item number in each sample data
    :return:
    """
    o, x, b = [], [], []
    tf.logging.info("Read dataset: " + path)

    for line in open(path):
        line_data = line.strip('\n').split(',')
        data_len = len(line_data)
        order = line_data[0]
        baseline = line_data[data_len - 1]
        inputs = line_data[1: data_len - 1]

        # Pad size data of items
        inputs = utils.items_size_padding(inputs, max_length)

        x.append(np.array(inputs, dtype=np.float32).reshape([-1, 3]))
        b.append(np.array(baseline, dtype=np.int32))
        o.append(np.array(order, dtype=np.int32))
    return o, x, b


class BinPackingDataLoader(object):
    def __init__(self, config):
        self.config = config

        self.task = config.task.lower()
        self.batch_size = config.batch_size
        self.min_length = config.min_data_length
        self.max_length = config.max_data_length

        self.is_train = config.is_train
        self.random_seed = config.random_seed

        self.data_num = dict()
        self.data_num['train'] = config.train_num
        self.data_num['test'] = config.test_num

        self.data_dir = config.data_dir
        self.task_name = "{}_({},{})".format(
            self.task, self.min_length, self.max_length)

        # Use self.data to store sample data from file
        # self.coord is tf.train.Coordinator
        # self.threads is multiple threads to push data into queue
        # self.order_ops is order data
        # self.input_ops is input data that is pushed into queue
        # self.target_ops is labelled data
        # self.queue_ops is the operation that pulls data from queue
        # self.enqueue_ops is the operation that pushes data into queue
        # self.o, self.x, self.b, self.seq_length is the batch data that are pulled from queue
        self.data = None
        self.coord = None
        self.threads = None
        self.order_ops, self.input_ops, self.target_ops = None, None, None
        self.queue_ops, self.enqueue_ops = None, None
        self.o, self.x, self.b, self.seq_length = None, None, None, None

        paths = {'train': 'data/drl_binpacking_baseline_train_la.txt',
                 'test': 'data/drl_binpacking_baseline_test_la.txt'}
        if len(paths) != 0:
            for name, path in paths.items():
                self.read_zip_and_update_data(path, name)

        # Create queue of training data and test data
        self._create_input_queue()

    def _create_input_queue(self):
        self.order_ops, self.input_ops, self.target_ops = {}, {}, {}
        self.queue_ops, self.enqueue_ops = {}, {}
        self.o, self.x, self.b, self.seq_length = {}, {}, {}, {}

        name = 'train'
        self.order_ops[name] = tf.placeholder(tf.int32, shape=[])
        self.input_ops[name] = tf.placeholder(tf.float32, shape=[None, None])
        self.target_ops[name] = tf.placeholder(tf.int32, shape=[])

        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * self.batch_size

        # Create a RandomShuffleQueue that pulls data from queue randomly
        self.queue_ops[name] = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            dtypes=[tf.int32, tf.float32, tf.int32],
            shapes=[[], [self.max_length, 3, ], []],
            seed=self.random_seed,
            name="random_queue_{}".format(name))

        # Define the operation that pushes data into queue
        self.enqueue_ops[name] = \
            self.queue_ops[name].enqueue([self.order_ops[name], self.input_ops[name], self.target_ops[name]])

        orders, inputs, baselines = self.queue_ops[name].dequeue()

        # Get sequence length (item number)
        seq_length = tf.shape(inputs)[0]

        # Create batch data using tf.train.batch
        self.o[name], self.x[name], self.b[name], self.seq_length[name] = \
            tf.train.batch(
                [orders, inputs, baselines, seq_length],
                batch_size=self.batch_size,
                capacity=capacity,
                dynamic_pad=True,
                name="batch_and_pad")

        name = 'test'
        self.order_ops[name] = tf.placeholder(tf.int32, shape=[])
        self.input_ops[name] = tf.placeholder(tf.float32, shape=[None, None])
        self.target_ops[name] = tf.placeholder(tf.int32, shape=[])

        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * 100000

        # Create a FIFOQueue that pulls data from a queue one by one
        self.queue_ops[name] = tf.FIFOQueue(
            capacity=capacity,
            dtypes=[tf.int32, tf.float32, tf.int32],
            shapes=[[], [self.max_length, 3, ], []],
            name="fifo_queue_{}".format(name))

        self.enqueue_ops[name] = \
            self.queue_ops[name].enqueue([self.order_ops[name], self.input_ops[name], self.target_ops[name]])

        orders, inputs, baselines = self.queue_ops[name].dequeue()

        seq_length = tf.shape(inputs)[0]

        self.o[name], self.x[name], self.b[name], self.seq_length[name] = \
            tf.train.batch(
                [orders, inputs, baselines, seq_length],
                batch_size=self.batch_size,
                capacity=capacity,
                dynamic_pad=True,
                name="batch_and_pad")

    def run_input_queue(self, sess):
        """
        Use multiple threads to push data into queue
        :param sess: current session
        :return:
        """
        self.threads = []
        self.coord = tf.train.Coordinator()

        for name in self.data_num.keys():
            def load_and_enqueue(sess, name, order_ops, input_ops, target_ops, coord):
                idx = 0

                # Get data from data['train'] or data['test'] until the thread is stopped
                while not coord.should_stop():
                    feed_dict = {
                        order_ops[name]: self.data[name].o[idx],
                        input_ops[name]: self.data[name].x[idx],
                        target_ops[name]: self.data[name].b[idx]
                    }
                    sess.run(self.enqueue_ops[name], feed_dict=feed_dict)
                    idx = idx + 1 if idx + 1 <= len(self.data[name].x) - 1 else 0

            args = (sess, name, self.order_ops, self.input_ops, self.target_ops, self.coord)
            t = threading.Thread(target=load_and_enqueue, args=args)
            t.start()
            self.threads.append(t)
            tf.logging.info("Thread for [{}] start".format(name))

    def stop_input_queue(self):
        """
        Stop the threads
        :return:
        """
        self.coord.request_stop()
        self.coord.join(self.threads, stop_grace_period_secs=200)
        tf.logging.info("All threads stopped")

    def read_zip_and_update_data(self, path, name):
        """
        :param path: path of data file
        :param name: name of data file (train or test)
        :return:
        """
        o_list, x_list, b_list = read_dataset(path, self.max_length)

        o = np.zeros([len(o_list), ], dtype=np.int32)
        x = np.zeros([len(x_list), self.max_length, 3], dtype=np.float32)
        b = np.zeros([len(b_list), ], dtype=np.int32)

        for idx, (os, nodes, res) in enumerate(zip(o_list, x_list, b_list)):
            o[idx] = os
            x[idx, :len(nodes)] = nodes
            b[idx] = res

        if self.data is None:
            self.data = {}

        self.data[name] = binpacking(o=o, x=x, b=b, name=name)

    def read_zip_and_update_data_new(self, path, name):
        """
        Another method to read and update data
        :param path: path of data file
        :param name: name of data file（train or test）
        :return:
        """
        tf.logging.info("Read dataset")

        o, x, b = [], [], []
        for line in open(path):
            line = line.strip('\n')
            line_data = line.split(',')

            # Check if the length of data is valid
            if len(line_data) != 32:
                continue

            o_data = line_data[0]
            x_data = []
            for i in range(10):
                x_data.append(line_data[i * 3 + 1: (i + 1) * 3 + 1])

            b_data = line_data[31]

            o.append(o_data)
            x.append(x_data)
            b.append(b_data)

        if self.data is None:
            self.data = {}

        self.data[name] = binpacking(o=np.array(o, dtype=np.float32), x=np.array(x, dtype=np.float32),
                                     b=np.array(b, dtype=np.float32), name=name)
