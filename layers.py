# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.util import nest
from tensorflow.contrib.layers.python.layers import utils

try:
    LSTMCell = rnn.LSTMCell
    dynamic_rnn_decoder = seq2seq.dynamic_rnn_decoder
    simple_decoder_fn_train = seq2seq.simple_decoder_fn_train
except:
    LSTMCell = tf.contrib.rnn.LSTMCell
    dynamic_rnn_decoder = tf.contrib.seq2seq.dynamic_rnn_decoder
    simple_decoder_fn_train = tf.contrib.seq2seq.simple_decoder_fn_train

try:
    smart_cond = utils.smart_cond
except:
    smart_cond = tf.contrib.layers.utils.smart_cond


def decoder_rnn(cell, inputs,
                enc_outputs, enc_final_states,
                seq_length, hidden_dim,
                batch_size, initializer=None,
                max_length=None):
    """
    Self-defined decoder_rnn
    :param cell: network cell
    :param inputs: input data
    :param enc_outputs: the output of encoder
    :param enc_final_states: the final states of encoder
    :param seq_length: the length of sequence
    :param hidden_dim: the dimension size of hidden layer
    :param batch_size: the size of batch data
    :param initializer: the method to initialize data
    :param max_length: the maximum length of sequence
    :return:
    """
    with tf.variable_scope("decoder_rnn") as scope:

        def attention(ref, query, tabu_list, with_softmax, scope="attention"):
            # Define attention mechanism
            with tf.variable_scope(scope):
                # Define variable
                # Shape of W_ref: [1, hidden_dim, hidden_dim]
                # Shape of W_q: [hidden_dim, hidden_dim]
                # Shape of v: [hidden_dim]
                W_ref = tf.get_variable(
                    "W_ref", [1, hidden_dim, hidden_dim], initializer=initializer)
                W_q = tf.get_variable(
                    "W_q", [hidden_dim, hidden_dim], initializer=initializer)
                v = tf.get_variable(
                    "v", [hidden_dim], initializer=initializer)

                # Use conv1d to encode ref
                encoded_ref = tf.nn.conv1d(ref, W_ref, 1, "VALID", name="encoded_ref")
                encoded_query = tf.expand_dims(tf.matmul(query, W_q, name="encoded_query"), 1)

                scores = tf.reduce_sum(v * tf.tanh(encoded_ref + encoded_query), [-1])
                scores = adjust_scores(scores, tabu_list)
                if with_softmax:
                    return tf.nn.softmax(scores)
                else:
                    return scores

        def adjust_scores(scores, tabu_list):
            """
            If a item has been fixed in a sequence, then adjust the corresponding score to negative infinity
            :param scores: Original scores
            :param tabu_list: The items that have been fixed in a sequence
            :return: Adjusted scores
            """
            tabu_matrix = (tf.ones(scores.get_shape()) - tabu_list) * tf.convert_to_tensor([1e15])
            adjusted_scores = scores - tabu_matrix

            return adjusted_scores

        def glimpse(ref, query, tabu_list, scope="glimpse"):
            # Define Glimpse mechanism
            p = attention(ref, query, tabu_list, with_softmax=True, scope=scope)
            alignments = tf.expand_dims(p, 2)

            return tf.reduce_sum(alignments * ref, [1])

        def output_fn(ref, query, tabu_list):
            if query is None:
                return tf.zeros([max_length], tf.float32)
            else:
                query = glimpse(ref, query, tabu_list, "glimpse_{}".format(1))
                return attention(ref, query, tabu_list, with_softmax=False, scope="attention")

        # Get the input of next cell
        def input_fn(sampled_idx):
            return tf.gather_nd(enc_outputs, index_matrix_to_pairs_new(sampled_idx))

        def update_context_state(original_context_state, idx):
            """
            After a item is selected, update context state
            :param original_context_state: Original context_state
            :param idx: the index of selected item
            :return: updated context_state
            """
            original_context_state = original_context_state
            idx = index_matrix_to_pairs_new(idx)
            idx = tf.cast(idx, tf.int64)
            matrix_index = tf.SparseTensor(idx, tf.ones(batch_size), [batch_size, max_length])
            matrix_index = tf.sparse_tensor_to_dense(matrix_index)
            new_context_state = original_context_state - tf.cast(matrix_index, tf.float32)
            return new_context_state

        maximum_length = tf.convert_to_tensor(max_length, tf.int32)

        def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
            cell_output = output_fn(enc_outputs, cell_output, context_state)
            if cell_state is None:
                cell_state = enc_final_states
                next_input = cell_input
                done = tf.zeros([batch_size, ], dtype=tf.bool)
                next_context_state = tf.ones([batch_size, max_length])
            else:
                sampled_idx = tf.argmax(cell_output, 1)
                next_context_state = update_context_state(context_state, sampled_idx)
                next_input = input_fn(sampled_idx)

            done = tf.cond(tf.greater_equal(time, maximum_length),
                lambda: tf.ones([batch_size, ], dtype=tf.bool),
                lambda: tf.zeros([batch_size, ], dtype=tf.bool))
            return (done, cell_state, next_input, cell_output, next_context_state)

        outputs, final_state, final_context_state = \
            dynamic_rnn_decoder(cell, decoder_fn, inputs=inputs,
                                sequence_length=seq_length, scope=scope)

        return outputs, final_state, final_context_state


def trainable_initial_state(batch_size, state_size,
                            initializer=None, name="initial_state"):
    flat_state_size = nest.flatten(state_size)

    if not initializer:
        flat_initializer = tuple(tf.zeros_initializer for _ in flat_state_size)
    else:
        flat_initializer = tuple(tf.zeros_initializer for _ in flat_state_size)

    names = ["{}_{}".format(name, i) for i in xrange(len(flat_state_size))]
    tiled_states = []

    for name, size, init in zip(names, flat_state_size, flat_initializer):
        shape_with_batch_dim = [1, size]
        # Create initial state variable, shape: [1, size], and the initial value is 0
        initial_state_variable = tf.get_variable(
            name, shape=shape_with_batch_dim, initializer=init())

        # Create tiled state variable, shape: [batch_size, 1]
        tiled_state = tf.tile(initial_state_variable,
                              [batch_size, 1], name=(name + "_tiled"))
        tiled_states.append(tiled_state)

    # Pack the tiled states
    return nest.pack_sequence_as(structure=state_size,
                                 flat_sequence=tiled_states)


def index_matrix_to_pairs(index_matrix):
    """
    Convert matrix into pairs
    [[3, 1, 2], [2, 3, 1]] -> [[[0, 3], [0, 1], [0, 2]], [[1, 2], [1, 3], [1, 1]]]
    :param index_matrix:
    :return:
    """
    replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
    rank = len(index_matrix.get_shape())
    if rank == 2:
        replicated_first_indices = tf.tile(
            tf.expand_dims(replicated_first_indices, dim=1),
            [1, tf.shape(index_matrix)[1]])
    return tf.stack([replicated_first_indices, index_matrix], axis=rank)


def index_matrix_to_pairs_new(index_matrix):
    """
    Another method to convert matrix into pairs
    :param index_matrix:
    :return:
    """
    index_matrix = tf.cast(index_matrix, tf.int32)
    first_dimension = index_matrix.get_shape()[0]
    replicated_first_indices = tf.convert_to_tensor([i for i in range(first_dimension)])
    rank = len(index_matrix.get_shape())

    if rank == 2:
        second_dimension = index_matrix.get_shape()[1]
        replicated_first_indices = tf.convert_to_tensor([[i] * second_dimension for i in range(first_dimension)])

    return tf.stack([replicated_first_indices, index_matrix], axis=rank)
