# -*- coding: UTF-8 -*-
import tensorflow as tf

from tensorflow.python.util import nest


def index_matrix_to_pairs(index_matrix):
    """
    Expand index_matrix
    [[3, 1, 2], [1, 2, 3]] -> [[0, 0, 3], [0, 1, 1], [0, 2, 2], [1, 0, 1], [1, 1, 2], [1, 2, 3]]
    :param index_matrix: The original index_matrix
    :return:
    """
    index_matrix = tf.cast(index_matrix, tf.int32)
    matrix_shape = index_matrix.get_shape()
    first_dimension = matrix_shape[0].value
    second_dimension = matrix_shape[1].value

    replicated_first_indices = []
    replicated_second_indices = []

    for i in range(first_dimension):
        replicated_first_indices += [i] * second_dimension
        replicated_second_indices += [j for j in range(second_dimension)]

    replicated_first_indices = tf.convert_to_tensor(replicated_first_indices)
    replicated_second_indices = tf.convert_to_tensor(replicated_second_indices)

    index_matrix = tf.reshape(index_matrix, [first_dimension * second_dimension, ])

    return tf.stack([replicated_first_indices, replicated_second_indices, index_matrix], axis=1)


def index_matrix_to_pairs_two(index_matrix):
    """
    Another method to expand index_matrix
    [[3, 1, 2], [1, 2, 3]] -> [[[0, 0, 3], [0, 1, 1], [0, 2, 2]], [[1, 0, 1], [1, 1, 2], [1, 2, 3]]]
    :param index_matrix: The original index_matrix
    :return:
    """
    index_matrix = tf.cast(index_matrix, tf.int32)
    matrix_shape = index_matrix.get_shape()
    first_dimension = matrix_shape[0].value
    second_dimension = matrix_shape[1].value

    replicated_first_indices = tf.convert_to_tensor([[i] * second_dimension for i in range(first_dimension)])
    replicated_second_indices = tf.convert_to_tensor([[i for i in range(second_dimension)]
                                                      for _ in range(first_dimension)])

    return tf.stack([replicated_first_indices, replicated_second_indices, index_matrix], axis=2)


def index_matrix_to_pairs_three(index_matrix):
    """
    Another method to expand index_matrix
    [[3, 1, 2], [1, 2, 3]] -> [[[0, 3], [0, 1], [0, 2]], [[0, 1], [0, 2], [0, 3]]]
    :param index_matrix: The original index_matrix
    :return:
    """
    index_matrix = tf.cast(index_matrix, tf.int32)
    matrix_shape = index_matrix.get_shape()
    first_dimension = matrix_shape[0].value
    second_dimension = matrix_shape[1].value

    replicated_first_indices = tf.convert_to_tensor([[i] * second_dimension for i in range(first_dimension)])

    return tf.stack([replicated_first_indices, index_matrix], axis=2)


def update_context_state_beam_search(context_state, index):
    """
    Update context state after a item is selected
    :param context_state: current context state, shape: [batch_size, beam_size, max_length]
    :param index: the index of selected item，shape: [batch_size, beam_size]
    :return: updated context_state
    """
    context_state_shape = context_state.get_shape()
    batch_size = context_state_shape[0].value
    beam_size = context_state_shape[1].value
    max_length = context_state_shape[2].value
    index = index_matrix_to_pairs(index)
    index = tf.cast(index, tf.int64)

    matrix_index = tf.SparseTensor(index, tf.ones(batch_size * beam_size), [batch_size, beam_size, max_length])
    matrix_index = tf.sparse_tensor_to_dense(matrix_index)
    new_context_state = context_state - tf.cast(matrix_index, tf.float32)
    return new_context_state


def data_nest_map(func, data):
    """
    Convert data
    :param func: convert function
    :param data: The data needed to be converted
    :return:
    """
    if not nest.is_sequence(data):
        return func(data)
    flat = nest.flatten(data)
    return nest.pack_sequence_as(data, list(map(func, flat)))


def flat_batch_gather(flat_data, indices, batch_size=None, gather_data_size=None):
    """
    Select data based on indices
    :param flat_data: origin data set
    :param indices: indices of data to be selected
    :param batch_size: batch_size
    :param gather_data_size: number of data to be selected
    :return:
    """
    if batch_size is None:
        batch_size = indices.get_shape()[0].value
        if batch_size is None:
            batch_size = tf.shape(indices)[0]

    if gather_data_size is None:
        gather_data_size = flat_data.get_shape()[0].value
        if gather_data_size is None:
            gather_data_size = tf.shape(flat_data)[0] // batch_size
        else:
            gather_data_size = gather_data_size // batch_size

    indices_offsets = tf.reshape(tf.range(batch_size) * gather_data_size, [-1] + [1] * (len(indices.get_shape()) - 1))

    indices_into_flat = indices + tf.cast(indices_offsets, indices.dtype)
    flat_indices_into_flat = tf.reshape(indices_into_flat, [-1])

    return tf.gather(flat_data, flat_indices_into_flat)


def batch_gather(data, indices, batch_size=None, gather_data_size=None):
    """
    Select data based on indices
    :param data: original data set
    :param indices: indices of data to be selected
    :param batch_size: batch_size
    :param gather_data_size: number of data to be selected
    :return:
    """
    if batch_size is None:
        batch_size = data.get_shape()[0].merge_with(indices.get_shape()[0]).value
        if batch_size is None:
            batch_size = tf.shape(indices)[0]

    if gather_data_size is None:
        gather_data_size = data.get_shape()[1].value
        if gather_data_size is None:
            gather_data_size = tf.shape(data)[1]

    batch_size_times_options_size = batch_size * gather_data_size

    flat_data = tf.reshape(data, tf.concat([[batch_size_times_options_size], tf.shape(data)[2:]], 0))

    indices_offsets = tf.reshape(tf.range(batch_size) * gather_data_size, [-1] + [1] * (len(indices.get_shape()) - 1))
    indices_into_flat = indices + tf.cast(indices_offsets, indices.dtype)

    return tf.gather(flat_data, indices_into_flat)


class BeamSearchReplicatedCell(tf.contrib.rnn.LSTMCell):
    def __init__(self, cell, beam_size):
        self.cell = cell
        self.beam_size = beam_size

    def prepend_beam_size(self, element):
        """
        Expand the state and output of cell
        :param element: The data to be expanded
        :return:
        """
        return tf.TensorShape(self.beam_size).concatenate(element)

    @property
    def state_size(self):
        """
        Expand the state
        :return:
        """
        return data_nest_map(self.prepend_beam_size, self.cell.state_size)

    @property
    def output_size(self):
        """
        Expand the output
        :return:
        """
        return data_nest_map(self.prepend_beam_size, self.cell.output_size)

    def tile_tensor(self, state):
        """
        Tile state data
        :param state:
        :return:
        """
        if nest.is_sequence(state):
            return data_nest_map(
                lambda val: self.tile_tensor(val),
                state
            )

        if not isinstance(state, tf.Tensor):
            raise ValueError("Cell state should be a sequence or tensor")

        tensor = state

        tensor_shape = tensor.get_shape().with_rank_at_least(1)

        dynamic_tensor_shape = tf.unstack(tf.shape(tensor))

        res_tensor = tf.expand_dims(tensor, 1)
        res_tensor = tf.tile(res_tensor, [1, self.beam_size] + [1] * (tensor_shape.ndims - 1))
        res_tensor = tf.reshape(res_tensor, [-1, self.beam_size] + list(dynamic_tensor_shape[1:]))
        new_tensor_shape = tensor_shape[:1].concatenate(self.beam_size).concatenate(tensor_shape[1:])
        res_tensor.set_shape(new_tensor_shape)

        return res_tensor

    def __call__(self, inputs, state, scope=None):
        var_scope = scope or tf.get_variable_scope()

        flat_inputs = nest.flatten(inputs)
        flat_state = nest.flatten(state)

        flat_inputs_unstacked = list(zip(*[tf.unstack(tensor, num=self.beam_size, axis=1) for tensor in flat_inputs]))
        flat_state_unstacked = list(zip(*[tf.unstack(tensor, num=self.beam_size, axis=1) for tensor in flat_state]))

        flat_output_unstacked = []
        flat_next_state_unstacked = []
        output_sample = None
        next_state_sample = None

        for index, (inputs_k, state_k) in enumerate(zip(flat_inputs_unstacked, flat_state_unstacked)):

            inputs_k = nest.pack_sequence_as(inputs, inputs_k)
            state_k = nest.pack_sequence_as(state, state_k)

            if index == 0:
                output_k, next_state_k = self.cell(inputs_k, state_k, scope=scope)
            else:
                with tf.variable_scope(var_scope, reuse=True):
                    output_k, next_state_k = self.cell(inputs_k, state_k,
                                                       scope=var_scope if scope is not None else None)

            flat_output_unstacked.append(nest.flatten(output_k))
            flat_next_state_unstacked.append(nest.flatten(next_state_k))

            output_sample = output_k
            next_state_sample = next_state_k

        flat_output = [tf.stack(tensor, axis=1) for tensor in zip(*flat_output_unstacked)]
        flat_next_state = [tf.stack(tensor, axis=1) for tensor in zip(*flat_next_state_unstacked)]

        output = nest.pack_sequence_as(output_sample, flat_output)
        next_state = nest.pack_sequence_as(next_state_sample, flat_next_state)

        return output, next_state


def beam_search_decoder(cell, batch_size, beam_size, encoder_outputs, initial_state, initial_input, initializer,
                        hidden_dim, max_length, scope=None):

    with tf.variable_scope(scope or "BeamSearchDecoder") as var_scope:

        invalid_score = -1e18

        def adjust_scores(scores, tabu_list):
            """
            If a item is fixed in a sequence, adjust the score of the item to be negative infinity
            :param scores: the original score
            :param tabu_list: the list of items that have been fixed in a sequence
            :return: adjusted score
            """

            tabu_matrix = (tf.ones(scores.get_shape()) - tabu_list) * tf.convert_to_tensor([invalid_score])
            adjusted_scores = scores + tabu_matrix

            return adjusted_scores

        def attention(ref, query, tabu_list, with_softmax, attention_scope="attention"):
            """
            Define attention mechanism
            :param ref: encoder_outputs
            :param query: the output of current cell
            :param tabu_list: the list of items that have been fixed in a sequence
            :param with_softmax: whether the softmax is used
            :param attention_scope:
            :return:
            """
            with tf.variable_scope(attention_scope):
                w_ref = tf.get_variable(
                    "w_ref", [1, hidden_dim, hidden_dim], initializer=initializer)
                w_q = tf.get_variable(
                    "w_q", [hidden_dim, hidden_dim], initializer=initializer)
                v = tf.get_variable(
                    "v", [hidden_dim], initializer=initializer)

                encoded_ref = tf.nn.conv1d(ref, w_ref, 1, "VALID", name="encoded_ref")

                encoded_ref = tf.tile(tf.expand_dims(encoded_ref, 1), (1, beam_size, 1, 1))

                encoded_ref = tf.reshape(encoded_ref,
                                         (batch_size * beam_size, max_length, hidden_dim))

                query = tf.reshape(query, (batch_size * beam_size, hidden_dim))
                encoded_query = tf.expand_dims(tf.matmul(query, w_q, name="encoded_query"), 1)

                scores = tf.reduce_sum(v * tf.tanh(encoded_ref + encoded_query), [-1])
                scores = tf.reshape(scores, (batch_size, beam_size, max_length))
                scores = adjust_scores(scores, tabu_list)
                if with_softmax:
                    return tf.nn.softmax(scores)
                else:
                    return scores

        def glimpse(ref, query, tabu_list, glimpse_scope="glimpse"):
            """
            Define Glimpse mechanism
            :param ref: encoder_outputs
            :param query: the output of current cell
            :param tabu_list: the list of items that have been fixed in a sequence
            :param glimpse_scope:
            :return:
            """
            prob = attention(ref, query, tabu_list, with_softmax=True, attention_scope=glimpse_scope)

            alignments = tf.expand_dims(prob, 3)

            ref = tf.tile(tf.expand_dims(ref, 1), (1, beam_size, 1, 1))
            return tf.reduce_sum(alignments * ref, [2])

        def output_fn(ref, query, tabu_list):
            """
            :param ref: encoder_outputs，shape: [batch_size, max_length, hidden_dim]
            :param query: the output of current cell, shape: [batch_size, beam_size, hidden_dim]
            :param tabu_list: the list of items that have been fixed in a sequence
            :return:
            """
            if query is None:
                return tf.zeros([batch_size, beam_size, max_length], tf.float32)
            else:
                query = glimpse(ref, query, tabu_list, "glimpse_{}".format(1))

                return tf.log(tf.constant(1e-18) +
                              attention(ref, query, tabu_list, with_softmax=True, attention_scope="attention"))

        def input_fn(sampled_idx):
            """
            :param sampled_idx: the item that haven been selected, shape: [batch_size, beam_size]
            :return:
            """

            return batch_gather(encoder_outputs, sampled_idx)

        def beam_search_setup():
            """
            Setup the loop of beam search
            :return:
            """
            next_cell_state = initial_state
            next_input = initial_input

            best_symbols = tf.fill([batch_size, 0], tf.constant(-1, dtype=tf.int32))
            best_log_prob = tf.ones((batch_size,), dtype=tf.float32) * -float('inf')

            first_in_beam_mask = tf.equal(tf.range(batch_size * beam_size) % beam_size, 0)

            beam_symbols = tf.fill([batch_size * beam_size, 0], tf.constant(-1, dtype=tf.int32))
            beam_log_prob = tf.where(
                first_in_beam_mask,
                tf.fill([batch_size * beam_size], 0.0),
                tf.fill([batch_size * beam_size], invalid_score)
            )

            best_symbols._shape = tf.TensorShape((batch_size, None))
            best_log_prob._shape = tf.TensorShape((batch_size,))
            beam_symbols._shape = tf.TensorShape((batch_size * beam_size, None))
            beam_log_prob._shape = tf.TensorShape((batch_size * beam_size,))

            next_context_state = tf.ones([batch_size, beam_size, max_length])

            next_loop_state = (
                best_symbols,
                best_log_prob,
                beam_symbols,
                beam_log_prob,
                next_context_state,
            )

            cell_output = tf.zeros(cell.output_size)
            is_finished = tf.zeros([batch_size], dtype=tf.bool)

            return (is_finished, next_input, next_cell_state,
                    cell_output, next_loop_state)

        def beam_search_loop(time, cell_output, cell_state, loop_state):
            """
            the loop of beam search
            :param time:
            :param cell_output: output of cell
            :param cell_state: state of cell
            :param loop_state: state of loop
            :return:
            """
            (past_best_symbols, past_best_log_prob, past_beam_symbols, past_beam_log_prob, context_state) = loop_state

            log_prob = output_fn(encoder_outputs, cell_output, context_state)

            log_prob_batched = tf.reshape(log_prob + tf.expand_dims(tf.reshape(past_beam_log_prob,
                                                                               [batch_size, beam_size]),
                                                                    2),
                                          [batch_size, beam_size * max_length])

            beam_log_prob, indices = tf.nn.top_k(log_prob_batched, beam_size)
            beam_log_prob = tf.reshape(beam_log_prob, [-1])

            symbols = indices % max_length
            parent_beam_refs = indices // max_length

            symbols_history = flat_batch_gather(past_beam_symbols, parent_beam_refs, batch_size=batch_size,
                                                gather_data_size=beam_size)

            next_context_state = batch_gather(context_state, parent_beam_refs, batch_size=batch_size,
                                              gather_data_size=beam_size)
            next_context_state = update_context_state_beam_search(next_context_state, symbols)

            beam_symbols = tf.concat([symbols_history, tf.reshape(symbols, [-1, 1])], 1)

            next_cell_state = data_nest_map(
                lambda element: batch_gather(element, parent_beam_refs, batch_size=batch_size,
                                             gather_data_size=beam_size),
                cell_state
            )

            next_input = input_fn(tf.reshape(symbols, [-1, beam_size]))

            best_log_prob = tf.reduce_max(tf.reshape(beam_log_prob, [batch_size, beam_size]), 1)
            best_symbols_refs = tf.argmax(tf.reshape(beam_log_prob, [batch_size, beam_size]), 1)
            best_symbols = flat_batch_gather(beam_symbols, best_symbols_refs, batch_size=batch_size,
                                             gather_data_size=beam_size)

            is_finished = tf.cond(tf.greater_equal(time, max_length),
                                  lambda: tf.ones([batch_size, ], dtype=tf.bool),
                                  lambda: tf.zeros([batch_size, ], dtype=tf.bool))

            for tensor in list(nest.flatten(next_input)) + list(nest.flatten(next_cell_state)):
                tensor.set_shape(tf.TensorShape((batch_size, beam_size))
                                 .concatenate(tensor.get_shape()[2:]))

            for tensor in [best_symbols, best_log_prob, is_finished]:
                tensor.set_shape(tf.TensorShape((batch_size,)).concatenate(tensor.get_shape()[1:]))

            for tensor in [beam_symbols, beam_log_prob]:
                tensor.set_shape(tf.TensorShape((batch_size * beam_size,))
                                 .concatenate(tensor.get_shape()[1:]))

            next_loop_state = (
                best_symbols,
                best_log_prob,
                beam_symbols,
                beam_log_prob,
                next_context_state,
            )

            return (is_finished, next_input, next_cell_state,
                    cell_output, next_loop_state)

        def loop_fn(time, cell_output, cell_state, loop_state):
            """
            Define loop_fn based on beam_search_setup and beam_search_loop
            :param time:
            :param cell_output:
            :param cell_state:
            :param loop_state:
            :return:
            """
            if cell_output is None:
                return beam_search_setup()
            else:
                return beam_search_loop(time, cell_output, cell_state, loop_state)

        def decode_dense_res():
            """
            :return:
            """
            emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(cell, loop_fn, scope=var_scope)
            best_symbols, best_log_prob, beam_symbols, beam_log_prob, context_state = final_loop_state
            return best_symbols, best_log_prob, beam_symbols, beam_log_prob, context_state

        return decode_dense_res()
