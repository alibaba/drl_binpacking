# -*- coding: UTF-8 -*-
import tensorlayer as tl
from layers import *
from beam_search_decoder import *
from utils import get_java_class
from utils import cal_objective


class Model(object):
    def __init__(self, config,
                 orders, inputs, baselines, enc_seq_length, dec_seq_length,
                 reuse=False):
        self.task = config.task
        self.config = config

        # the dimension size of input data, such as: 3
        self.input_dim = config.input_dim
        # the dimension size of hidden layer, such as: 256
        self.hidden_dim = config.hidden_dim
        # the batch size of training data, such as: 128
        self.batch_size = config.batch_size

        # the maximum length of encoder sequence and decoder sequence
        self.max_enc_length = config.max_enc_length
        self.max_dec_length = config.max_dec_length
        self.seq_length = config.max_data_length

        self.input_keep_prob = config.input_keep_prob
        self.output_keep_prob = config.output_keep_prob

        self.is_beam_search_used = config.is_beam_search_used
        self.beam_size = config.beam_size

        self.init_first_decoder_input = config.init_first_decoder_input

        self.init_min_val = config.init_min_val
        self.init_max_val = config.init_max_val
        # User uniform distribution to initialize the variables
        self.initializer = \
            tf.random_uniform_initializer(self.init_min_val, self.init_max_val)

        # The start value, decay step and decay rate of learning rate
        self.lr_start = config.lr_start
        self.lr_decay_step = config.lr_decay_step
        self.lr_decay_rate = config.lr_decay_rate

        # The parameter used to clip gradient
        self.max_grad_norm = config.max_grad_norm

        # The Java class to calculate objective function value
        self.java_class = get_java_class()

        self.input_keep_prob_placeholder = tf.placeholder(tf.float32, shape=())
        self.output_keep_prob_placeholder = tf.placeholder(tf.float32, shape=())
        self.orders_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size, ))
        self.enc_inputs_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, 10, 3))
        self.baselines_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size,))
        self.enc_seq_length_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, ))
        self.dec_seq_length_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, ))
        self.adjusted_obj_value = tf.placeholder(tf.float32, shape=(self.batch_size, ))

        ##############
        # inputs
        ##############

        self.is_training = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool),
                                                       shape=(), name='is_training')

        self.orders, self.enc_inputs, self.baselines, self.enc_seq_length, self.dec_seq_length = \
            smart_cond(
                self.is_training,
                lambda: (orders['train'], inputs['train'], baselines['train'], enc_seq_length['train'],
                         dec_seq_length['train']),
                lambda: (orders['test'], inputs['test'], baselines['test'], enc_seq_length['test'],
                         dec_seq_length['test'])
            )

        self._build_model()
        self._build_steps()

        if not reuse:
            self._build_optim()

        self.train_summary = tf.summary.merge([
            tf.summary.scalar("train/total_loss", self.total_loss),
            tf.summary.scalar("train/lr", self.lr),
        ])

        self.test_summary = tf.summary.merge([
            tf.summary.scalar("test/total_loss", self.total_loss),
        ])

    def _build_steps(self):
        """
        Build training step and test step
        :return:
        """

        def train(sess, summary_writer):

            orders, enc_inputs, baselines, enc_seq_length, dec_seq_length = sess.run([self.orders, self.enc_inputs,
                                                                                      self.baselines,
                                                                                      self.enc_seq_length,
                                                                                      self.dec_seq_length],
                                                                                     feed_dict={self.is_training: True})

            fetch = dict()

            fetch['step'] = self.global_step
            fetch['dec_pred'] = self.dec_pred

            predict_result = sess.run(
                fetch, feed_dict={self.orders_placeholder: orders,
                                  self.baselines_placeholder: baselines,
                                  self.enc_inputs_placeholder: enc_inputs,
                                  self.enc_seq_length_placeholder: enc_seq_length,
                                  self.dec_seq_length_placeholder: dec_seq_length,
                                  self.input_keep_prob_placeholder: self.input_keep_prob,
                                  self.output_keep_prob_placeholder: self.output_keep_prob
                                  }
            )

            items_size = enc_inputs
            obj_value = cal_objective(self.java_class, items_size, predict_result['dec_pred'])
            adjusted_obj_value = obj_value - baselines

            fetches = {"step": self.global_step, "total_loss": self.total_loss, "optim": self.optim}

            if self.train_summary is not None:
                fetches['summary'] = self.train_summary

            feed_dict = {self.adjusted_obj_value: adjusted_obj_value,
                         self.orders_placeholder: orders,
                         self.baselines_placeholder: baselines,
                         self.enc_inputs_placeholder: enc_inputs,
                         self.enc_seq_length_placeholder: enc_seq_length,
                         self.dec_seq_length_placeholder: dec_seq_length,
                         self.input_keep_prob_placeholder: self.input_keep_prob,
                         self.output_keep_prob_placeholder: self.output_keep_prob,
                         }
            result = sess.run(fetches=fetches, feed_dict=feed_dict)

            if summary_writer is not None:
                summary_writer.add_summary(result['summary'], result['step'])
                summary_writer.flush()

            return {'orders': orders, 'result': result, 'total_loss': result['total_loss'],
                    'dec_pred': predict_result['dec_pred'], 'step': result['step'],
                    'baselines': baselines, 'obj_value': obj_value, 'adjusted_obj_value': adjusted_obj_value}

        def test(sess, summary_writer=None):
            orders, enc_inputs, baselines, enc_seq_length, dec_seq_length = sess.run([self.orders, self.enc_inputs,
                                                                                      self.baselines,
                                                                                      self.enc_seq_length,
                                                                                      self.dec_seq_length],
                                                                                     feed_dict={self.is_training: False})
            fetch = dict()
            fetch['step'] = self.global_step
            fetch['dec_pred'] = self.dec_pred
            fetch['dec_pred_prob'] = self.dec_pred_prob

            predict_result = sess.run(
                fetch, feed_dict={self.orders_placeholder: orders,
                                  self.baselines_placeholder: baselines,
                                  self.enc_inputs_placeholder: enc_inputs,
                                  self.enc_seq_length_placeholder: enc_seq_length,
                                  self.dec_seq_length_placeholder: dec_seq_length,
                                  self.input_keep_prob_placeholder: 1.0,
                                  self.output_keep_prob_placeholder: 1.0,
                                  })
            items_size = enc_inputs

            obj_value = cal_objective(self.java_class, items_size, predict_result['dec_pred'])
            adjusted_obj_value = obj_value - baselines
            fetches = {"step": self.global_step, "total_loss": self.total_loss}

            if self.test_summary is not None:
                fetches['summary'] = self.test_summary

            feed_dict = {self.adjusted_obj_value: adjusted_obj_value,
                         self.orders_placeholder: orders,
                         self.baselines_placeholder: baselines,
                         self.enc_inputs_placeholder: enc_inputs,
                         self.enc_seq_length_placeholder: enc_seq_length,
                         self.dec_seq_length_placeholder: dec_seq_length,
                         self.input_keep_prob_placeholder: 1.0,
                         self.output_keep_prob_placeholder: 1.0,
                         }
            result = sess.run(fetches=fetches, feed_dict=feed_dict)

            if summary_writer is not None:
                summary_writer.add_summary(result['summary'], result['step'])
                summary_writer.flush()

            return {'orders': orders, 'result': result, 'total_loss': result['total_loss'],
                    'dec_pred': predict_result['dec_pred'], 'step': result['step'],
                    'baselines': baselines, 'obj_value': obj_value, 'adjusted_obj_value': adjusted_obj_value}

        self.train = train
        self.test = test

    def _build_model(self):
        """
        Build model
        :return:
        """
        tf.logging.info("Create a model..")
        self.global_step = tf.Variable(0, trainable=False)

        # Create input_embed, shape: [1, 3, 256]
        self.input_embed = tf.get_variable(
            "input_embed", [1, self.input_dim, self.hidden_dim],
            initializer=self.initializer)
        batch_size = tf.shape(self.enc_inputs_placeholder)[0]
        with tf.variable_scope("encoder"):

            is_tensorlayer_used = False
            if is_tensorlayer_used:
                # Create input layer based on enc_inputs
                input_layer = tl.layers.InputLayer(self.enc_inputs_placeholder, name='input_layer')

                # Embedding input layer based on conv1d
                encoder_network = tl.layers.Conv1dLayer(
                    layer=input_layer,
                    shape=[1, self.input_dim, self.hidden_dim],
                    padding='VALID',
                    W_init=tf.random_uniform_initializer(self.init_min_val, self.init_max_val),
                    b_init=tf.random_uniform_initializer(self.init_min_val, self.init_max_val),
                    name='conv_layer'
                )

                encoder_network = tl.layers.RNNLayer(encoder_network,
                                                     cell_fn=LSTMCell,  # tf.nn.rnn_cell.BasicLSTMCell,
                                                     cell_init_args={'forget_bias': 0.0, 'state_is_tuple': True},
                                                     n_hidden=self.hidden_dim,
                                                     initializer=tf.random_uniform_initializer(self.init_min_val, self.init_max_val),
                                                     n_steps=self.seq_length,
                                                     return_last=False,
                                                     name='encoder_network')

                # Get the output and final state of encoder network
                self.enc_outputs = encoder_network.outputs
                self.enc_final_states = encoder_network.final_state
            else:
                self.embeded_enc_inputs = tf.nn.conv1d(
                    self.enc_inputs_placeholder, self.input_embed, 1, "VALID")

                # Create LSTMCell for encoder network
                self.enc_cell = LSTMCell(
                    self.hidden_dim,
                    initializer=self.initializer)

                # Add dropout
                self.enc_cell = tf.contrib.rnn.DropoutWrapper(self.enc_cell, self.input_keep_prob_placeholder,
                                                              self.output_keep_prob_placeholder)

                self.enc_init_state = trainable_initial_state(
                    self.batch_size, self.enc_cell.state_size)

                # Get the output and final state of encoder network
                self.enc_outputs, self.enc_final_states = tf.nn.dynamic_rnn(
                    self.enc_cell, self.embeded_enc_inputs,
                    self.enc_seq_length_placeholder, self.enc_init_state)

            # Use max, min, or average value as the first input of decoder network
            if self.init_first_decoder_input == 'avg':
                self.first_decoder_input = tf.reduce_mean(self.enc_outputs, axis=1, keep_dims=True)
            elif self.init_first_decoder_input == 'max':
                self.first_decoder_input = tf.reduce_max(self.enc_outputs, axis=1, keep_dims=True)
            elif self.init_first_decoder_input == 'min':
                self.first_decoder_input = tf.reduce_min(self.enc_outputs, axis=1, keep_dims=True)
            else:
                self.first_decoder_input = tf.expand_dims(trainable_initial_state(
                    self.batch_size, self.hidden_dim, name="first_decoder_input"), 1)

        with tf.variable_scope("decoder"):

            self.dec_cell = LSTMCell(
                self.hidden_dim,
                initializer=self.initializer)

            self.dec_cell = tf.contrib.rnn.DropoutWrapper(self.dec_cell, self.input_keep_prob_placeholder,
                                                          self.output_keep_prob_placeholder)

            if not self.is_beam_search_used:

                self.dec_pred_logits, _, _ = decoder_rnn(
                    self.dec_cell, self.first_decoder_input,
                    self.enc_outputs, self.enc_final_states,
                    self.dec_seq_length_placeholder, self.hidden_dim,
                    self.batch_size, initializer=self.initializer,
                    max_length=self.max_dec_length
                )

                # Get predict probability
                self.dec_pred_prob = tf.nn.softmax(
                    self.dec_pred_logits, -1, name="dec_pred_prob")

                self.dec_pred = tf.argmax(
                    self.dec_pred_logits, 2, name="dec_pred")

                self.max_prob = tf.reduce_max(self.dec_pred_prob, reduction_indices=2)
                self.max_prob_product = tf.reduce_prod(self.max_prob, reduction_indices=1)
                self.log_prob = tf.log(self.max_prob_product)

            else:
                self.dec_cell = BeamSearchReplicatedCell(self.dec_cell, self.beam_size)

                self.first_decoder_input = tf.reshape(self.first_decoder_input, (self.batch_size, self.hidden_dim))
                self.first_decoder_input = self.dec_cell.tile_tensor(self.first_decoder_input)

                self.enc_final_states = self.dec_cell.tile_tensor(self.enc_final_states)

                self.dec_pred, self.log_prob, _, _, _ = beam_search_decoder(
                    self.dec_cell, self.batch_size, self.beam_size, self.enc_outputs, self.enc_final_states,
                    self.first_decoder_input, self.initializer, self.hidden_dim, self.max_dec_length,
                    scope="BeamSearchDecoder"
                )

    def _build_optim(self):
        batch_loss = tf.reduce_mean(self.log_prob * self.adjusted_obj_value)

        tf.losses.add_loss(batch_loss)
        total_loss = tf.losses.get_total_loss()

        self.total_loss = total_loss

        self.lr = tf.train.exponential_decay(
            self.lr_start, self.global_step, self.lr_decay_step,
            self.lr_decay_rate, staircase=True, name="learning_rate")

        optimizer = tf.train.AdamOptimizer(self.lr)
        # tf.logging.info(optimizer.get_slot_names())
        if self.max_grad_norm:
            grads_and_vars = optimizer.compute_gradients(self.total_loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:

                    grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)

            # Update variable value by clipped gradients
            self.optim = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        else:
            self.optim = optimizer.minimize(self.total_loss, global_step=self.global_step)
