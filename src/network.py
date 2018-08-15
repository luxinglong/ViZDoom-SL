import tensorflow as tf
import numpy as np

from utils import variable_summaries

WEIGHT_DECAY=0.1

class NetworkModel(object): 
    def __init__(self, params):

        # feature configure parameter
        self.batch_size = params.batch_size
        self.learning_rate = params.learning_rate
        self.feature_img_height = params.feature_img_height
        self.feature_img_width = params.feature_img_width
        self.feature_img_channel = params.feature_img_channel
        self.feature_vec_length = params.feature_vec_length

        # control the framework
        self.n_features = 2
        self.n_actions = 29
        
        # lstm
        self.seq_len = params.seq_len
        self.basic_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=512)
        self.pre_state = self.basic_cell.zero_state(1, dtype=tf.float32)

    def build_train_graph(self, img, label_sc, label_gf, keep_prob):
        
        self.init_state = self.basic_cell.zero_state(tf.shape(img)[0], dtype=tf.float32)
        label_sc = tf.reshape(tf.one_hot(label_sc, self.n_actions), [-1, self.n_actions])
        img = tf.reshape(img, [-1, self.feature_img_height, self.feature_img_width, self.feature_img_channel])
        
        output_sc, output_gf, next_state = self._inference(img, keep_prob, pre_state = self.init_state)

        output_gf = tf.reshape(tf.slice(output_gf, [0, self.seq_len-1, 0], [-1, 1, -1]), [-1, self.n_features])
        output_sc = tf.reshape(tf.slice(output_sc, [0, self.seq_len-1, 0], [-1, 1, -1]), [-1, self.n_actions])
        loss_sc, loss_gf = self._calculate_loss(output_sc, label_sc, output_gf, label_gf)

        output_gf = tf.nn.sigmoid(output_gf)
        output_sc = tf.nn.softmax(output_sc)
        accuracy_sc, accuracy_gf = self._calculate_accuracy(output_sc, label_sc, output_gf, label_gf)
        
        with tf.name_scope("train"):
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_sc+1.*loss_gf)

        return train_op, output_sc, loss_sc, accuracy_sc, output_gf, loss_gf, accuracy_gf 

    
    def _inference(self, feature_img, keep_prob, pre_state):
        
        input_channel = self.feature_img_channel

        with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
            conv1_kernel = self._conv_weight_variable(shape=[8, 8, input_channel, 32], name="conv1_kernel")
            conv1_bias = self._bias_variable(shape=[32], name="conv1_bias")
            conv1_result = tf.nn.relu((tf.nn.conv2d(feature_img, conv1_kernel, strides=[1, 4, 4, 1], padding="VALID") + conv1_bias), name="conv1_result")
        print('conv1_result shape: ', conv1_result.shape)

        with tf.variable_scope("conv2", reuse=tf.AUTO_REUSE):
            conv2_kernel = self._conv_weight_variable(shape=[4, 4, 32, 64], name="conv2_kernel")
            conv2_bias = self._bias_variable(shape=[64], name="conv2_bias")
            conv2_result = tf.nn.relu((tf.nn.conv2d(conv1_result, conv2_kernel, strides=[1, 2, 2, 1], padding="VALID") + conv2_bias), name="conv2_result")
        print('conv2_result shape: ', conv2_result.shape)

        with tf.variable_scope("flatten", reuse=tf.AUTO_REUSE):
            conv2_flatten = tf.layers.flatten(conv2_result, name="flatten")
        print('conv2_flatten shape: ', conv2_flatten)

        with tf.variable_scope("game_feature_fc1", reuse=tf.AUTO_REUSE):
            fc_gf_1_weight = self._fc_weight_variable(shape=[4608, 512], name="fc_gf_1_weight")
            fc_gf_1_bias = self._bias_variable(shape=[512], name="fc_gf_1_bias")
            fc_gf_1_dropout = tf.nn.dropout(conv2_flatten, keep_prob)
            fc_gf_1_result = tf.nn.relu((tf.matmul(fc_gf_1_dropout, fc_gf_1_weight) + fc_gf_1_bias), name="fc_gf_1_result")
        print('fc_gf_1_result shape: ', fc_gf_1_result.shape)

        with tf.variable_scope("game_feature_fc2", reuse=tf.AUTO_REUSE):
            fc_gf_2_weight = self._fc_weight_variable(shape=[512, self.n_features], name='fc_gf_2_weight')
            fc_gf_2_bias = self._bias_variable(shape=[self.n_features], name="fc_gf_2_bias")
            fc_gf_2_dropout = tf.nn.dropout(fc_gf_1_result, keep_prob)
            output_gf = tf.identity((tf.matmul(fc_gf_2_dropout, fc_gf_2_weight) + fc_gf_2_bias), name="fc_gf_2_result")
        output_gf = tf.reshape(output_gf, [-1, self.seq_len, self.n_features])
        print('output_gf shape: ', output_gf.shape)

        with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
            conv2_flatten_reshape = tf.reshape(conv2_flatten, [-1, self.seq_len, 4608], name="conv2_flatten_reshape")
            cell_dr = tf.nn.rnn_cell.DropoutWrapper(self.basic_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
            lstm_result, next_state = tf.nn.dynamic_rnn(cell=cell_dr, inputs=conv2_flatten_reshape, initial_state=pre_state, dtype=tf.float32, time_major=False)
        print('lstm_result shape: ', lstm_result.shape)
        
        lstm_result = tf.reshape(lstm_result, [-1, 512])

        with tf.variable_scope("lstm_sc", reuse=tf.AUTO_REUSE):
            lstm_sc_weight = self._fc_weight_variable(shape=[512, self.n_actions], name="lstm_sc_weight")
            lstm_sc_bias = self._bias_variable(shape=[self.n_actions], name="lstm_sc_bias")
            lstm_sc_dropout = tf.nn.dropout(lstm_result, keep_prob)
            lstm_sc_result = tf.identity((tf.matmul(lstm_sc_dropout, lstm_sc_weight) + lstm_sc_bias), name="lstm_sc_result")
        print('lstm_sc_result shape: ', lstm_sc_result.shape)

        output_sc = tf.reshape(lstm_sc_result, [-1, self.seq_len, self.n_actions])
        print('output_sc shape: ', output_sc.shape)

        return output_sc, output_gf, next_state
    
    
    def _calculate_loss(self, output_sc, label_sc, output_gf, label_gf):
        label_sc = tf.identity(label_sc, name="label_sc")
        label_gf = tf.identity(label_gf, name="label_gf")
        with tf.name_scope("loss"):
            # action score loss, softmax_cross_entropy
            loss_sc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_sc, logits=output_sc))
            # game features loss, sigmoid_cross_entropy
            loss_gf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_gf, logits=output_gf))
        return loss_sc, loss_gf

    
    def _calculate_accuracy(self, output_sc, label_sc, output_gf, label_gf):
        label_sc = tf.identity(label_sc, name="label_sc")
        label_gf = tf.identity(label_gf, name="label_gf")
        with tf.name_scope('accuracy'):
            # action score accuracy
            correct_prediction_sc = tf.equal(tf.argmax(output_sc, 1), tf.argmax(label_sc, 1))
            accuracy_sc = tf.reduce_mean(tf.cast(correct_prediction_sc, tf.float32))
            # game features accuracy    
            label_gf = tf.cast(label_gf, tf.int32)
            correct_prediction_gf = tf.equal(tf.cast(tf.round(output_gf), tf.int32), label_gf)
            accuracy_gf = tf.reduce_mean(tf.cast(correct_prediction_gf, tf.float32))
        return accuracy_sc, accuracy_gf

    
    def _conv_weight_variable(self, shape, name):
        regularizer = tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        return tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer)

    
    def _fc_weight_variable(self, shape, name):
        regularizer = tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer)

    
    def _bias_variable(self, shape, name):
        regularizer = tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)
        initializer = tf.constant_initializer(0.0)
        return tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer)
