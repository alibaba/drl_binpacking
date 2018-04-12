# -*- coding: UTF-8 -*-
import os
import json
import logging
from datetime import datetime
import tensorflow as tf
import tensorflow.contrib.slim as slim
import jpype as jp
import numpy as np


def prepare_dirs_and_logger(config):
    # Get a logger named "tensorflow"
    logger = logging.getLogger("tensorflow")

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()

    # Set the format of log
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set the level of log (NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(tf.logging.INFO)

    if config.load_path:
        if config.load_path.startswith(config.task):
            config.model_name = config.load_path
        else:
            config.model_name = "{}_{}".format(config.task, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.task, get_time())

    config.model_dir = os.path.join(config.log_dir, config.model_name)

    # Check and create logger folder, data folder and model folder
    for path in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def get_time():
    """
    Get current time, format: "2017-01-01_00:00:00"
    :return: The string of current time
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def show_all_variables():
    """
    Show the information of training variables
    :return:
    """
    # Get all training variables. The return data is a list, and the element is training variable.
    model_vars = tf.trainable_variables()

    # Print name, data type, shape, size, bytes number of variable
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def save_config(model_dir, config):
    """
    Save configuration data into file
    :param model_dir: The folder path of model
    :param config: Configuration data
    :return:
    """
    param_path = os.path.join(model_dir, "params.json")

    tf.logging.info("MODEL dir: %s" % model_dir)
    tf.logging.info("PARAM path: %s" % param_path)

    # Get configuration data, the key is configuration name  and the value is configuration value
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def start_jvm(jar_path):
    """
    :param jar_path: The path of jar file
    :return:
    """
    jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path={}".format(jar_path))


def get_java_class():
    """
    Get the Java class to calculate objective function value
    :return:
    """
    return jp.JPackage("com.taobao.idad.solver.binpack").ObjectiveValueCalculator


def shutdown_jvm():
    """
    :return:
    """
    jp.shutdownJVM()


def cal_objective(java_class, items_size, res_seq_matrix):
    """
    Calculate the objective function value of a item sequence
    :param java_class: The java class to calculate objective function value
    :param items_size: The size of items (length, width, height)
    :param res_seq_matrix: The sequence of items
    :return: Objective function value
    """
    # Get the size data of sorted times
    matrix_shape = res_seq_matrix.shape
    first_dimension, second_dimension = matrix_shape[0], matrix_shape[1]

    items_size = items_size.astype(np.int32)
    ordered_items_size = np.zeros(items_size.shape, dtype=np.int32)

    for i in range(first_dimension):
        for j in range(second_dimension):
            ordered_items_size[i][j] = items_size[i][res_seq_matrix[i][j]]

    ordered_items_size_data = ordered_items_size.tolist()
    ordered_items_size_data = jp.JArray(jp.JInt, 3)(ordered_items_size_data)

    obj_val = java_class.calObjectiveValue(ordered_items_size_data)
    obj_val = list(obj_val)
    return np.array(obj_val)


def items_size_padding(origin_items_size, max_items_num):
    """
    Padding item list (Use the item with length 0, height 0 and width 0)
    :param origin_items_size: The size of original items
    :param max_items_num: Max number of items
    :return: The size of items after padding
    """
    if len(origin_items_size) >= 3 * max_items_num:
        return origin_items_size

    return origin_items_size + [0.0] * (3 * max_items_num - len(origin_items_size))