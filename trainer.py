# -*- coding: UTF-8 -*-
import tensorflow as tf

from model import Model
from utils import show_all_variables
from read_data import BinPackingDataLoader


class Trainer(object):
    def __init__(self, config):
        self.config = config

        self.task = config.task
        self.model_dir = config.model_dir
        self.gpu_memory_fraction = config.gpu_memory_fraction
        self.init_first_decoder_input = config.init_first_decoder_input

        self.log_step = config.log_step
        self.max_step = config.max_step
        self.num_log_samples = config.num_log_samples
        self.checkpoint_secs = config.checkpoint_secs

        if config.task.lower().startswith('binpacking'):
            # Load train and test data
            self.data_loader = BinPackingDataLoader(config)
        else:
            raise Exception("[!] Unknown task: {}".format(config.task))

        # Build model based on data and config
        self.model = Model(
            config,
            orders=self.data_loader.o,
            inputs=self.data_loader.x,
            baselines=self.data_loader.b,
            enc_seq_length=self.data_loader.seq_length,
            dec_seq_length=self.data_loader.seq_length,
        )

        self.build_session()
        show_all_variables()

    def build_session(self):

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                 is_chief=True,
                                 saver=self.saver,
                                 summary_op=None,
                                 summary_writer=self.summary_writer,
                                 save_summaries_secs=300,
                                 save_model_secs=self.checkpoint_secs,
                                 global_step=self.model.global_step)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=self.gpu_memory_fraction,
            allow_growth=True)

        # allow_soft_placement=true indicates that tensorflow can choose a device automatically
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

    def train(self):
        """
        The main process of training
        :return:
        """
        tf.logging.info("Training starts...")
        # Push data into queue
        self.data_loader.run_input_queue(self.sess)

        summary_writer = None

        log_diff = []
        for k in range(self.max_step):
            print("Step: " + str(k))
            result = self.model.train(self.sess, summary_writer)
            sum_baseline = sum(result['baselines'])
            sum_obj_value = sum(result['obj_value'])
            sum_adjust_obj_value = sum_baseline - sum_obj_value
            ratio = sum_adjust_obj_value/sum_baseline
            log_diff_info = "_".join([format(k), format(sum_baseline), format(sum_obj_value), format(ratio)])
            log_diff.append(log_diff_info)

            # if result['step'] % self.log_step == 0:
            #     self._test(self.summary_writer)

            # summary_writer = self._get_summary_writer(result)

        self.data_save(log_diff, 'log.txt')

        tot = []

        for k in range(1000):
            result = self.model.test(self.sess, summary_writer)
            for idx in range(127):
                order_id = format(result['orders'][idx])
                dec_pred = format(result['dec_pred'][idx])
                baseline = format(result['baselines'][idx])
                obj_value = format(result['obj_value'][idx])
                adjusted_obj_value = format(result['adjusted_obj_value'][idx])
                res = "_".join([order_id, dec_pred, baseline, obj_value, adjusted_obj_value])
                tot.append(res)
                # tf.logging.info(res)
        self.data_save(tot, 'train_save.txt')

        self.data_loader.stop_input_queue()

    def test(self):
        """
        The main process of test
        :return:
        """
        tf.logging.info("Test Starts...")
        self.data_loader.run_input_queue(self.sess)

        tot = self.return_result(None)
        self.data_save(tot, 'test_save.txt')
        self.data_loader.stop_input_queue()

    def data_save(self, tot, path):
        output = open(path, 'w')
        for i in tot:
            # if data is a string, write into file directly. Otherwise, convert data to string, then write into file
            if isinstance(i, str):
                output.write(i)
                output.write('\n')
            else:
                j = ",".join(map(str, i))
                output.write(j)
                output.write('\n')

    def return_result(self, summary_writer):
        tot = []

        result = self.model.test(self.sess, summary_writer)
        for idx in range(1000):
            dec_pred = format(result['dec_pred'][idx])
            baseline = format(result['baselines'][idx])
            obj_value = format(result['obj_value'][idx])
            adjusted_obj_value = format(result['adjusted_obj_value'][idx])

            res = "_".join([dec_pred, baseline, obj_value, adjusted_obj_value])

            tot.append(res)
            tf.logging.info(res)
        return tot

    def _test(self, summary_writer):
        result = self.model.test(self.sess, summary_writer)

        tf.logging.info("")
        tf.logging.info("test loss: {}".format(result['total_loss']))
        for idx in range(self.num_log_samples):
            pred = result['dec_pred'][idx]
            baseline = result['baselines'][idx]
            obj_value = result['obj_value'][idx]
            adjusted_obj_value = result['adjusted_obj_value'][idx]
            tf.logging.info("test pred: {}".format(pred))
            tf.logging.info("test baseline: {}".format(baseline))
            tf.logging.info("test obj_value: {}".format(obj_value))
            tf.logging.info("test adjusted_obj_value: {}".format(adjusted_obj_value))

        if summary_writer:
            summary_writer.add_summary(result['step'])

    def _get_summary_writer(self, result):
        """
        :param result:
        :return:
        """
        if result['step'] % self.log_step == 0:
            return self.summary_writer
        else:
            return None
