# -*- coding: UTF-8 -*-
import os
import sys
import tensorflow as tf

from trainer import Trainer
from config import get_config
from utils import prepare_dirs_and_logger, save_config, start_jvm, shutdown_jvm

config = None


def main(_):
    """
    The main process of train and test
    :param _:
    :return:
    """
    prepare_dirs_and_logger(config)
    if not config.task.lower().startswith('binpacking'):
        raise Exception("[!] Task should starts with binpacking")

    if config.max_enc_length is None:
        config.max_enc_length = config.max_data_length
    if config.max_dec_length is None:
        config.max_dec_length = config.max_data_length

    tf.set_random_seed(config.random_seed)

    # A jar is used to calculate the objective function value, so start the JVM first.
    path = os.getcwd()
    jar_path = path + "/idad-solver-binpacking/idad-solver-binpacking_least_area.jar"
    start_jvm(jar_path)

    trainer = Trainer(config)
    save_config(config.model_dir, config)

    if config.is_train:
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

    tf.logging.info("Run finished.")

    shutdown_jvm()


if __name__ == "__main__":
    config, unparsed = get_config()
    # The entry point of tensorflow program
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)