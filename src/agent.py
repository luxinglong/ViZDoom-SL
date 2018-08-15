import os
import h5py
import numpy as np
import tensorflow as tf
from data_iter import next_batch, train_test_data
from network import NetworkModel


class Agent:
    
    def __init__(self, sess, params):
        '''Agent - powered by neural network, can trian and infer
        '''
        self.sess = sess
        self.params = params
        self.test_file_path = '/.firejobdata/data1/banlu/ViZDoom-SL/ViZDoomSL_baseline_lstm/data/dataset-1-test.hdf5'
        self.test_index = 0
        self.model = NetworkModel(params=params)
        self.build_model(self.params.log_dir)
    
    def choose_action(self, screen):
        predict_action, output_gf = self.sess.run([self.output_sc, self.output_gf], feed_dict={self.img: screen, self.keep_prob: 1})
        print predict_action
        action_id = int(np.argmax(predict_action, 1)[0])
        return action_id, output_gf

    def build_model(self, log_dir):
        self.img = tf.placeholder(tf.float32, shape=[None, self.params.seq_len, 60, 108, 3], name="img")
        # the latest action and game feature label
        self.label_sc = tf.placeholder(tf.int32, shape=[None, 1], name="label_sc")
        self.label_gf = tf.placeholder(tf.float32, shape=[None, 2], name="label_gf")
        self.keep_prob = tf.placeholder(tf.float32)

        train_op, output_sc, loss_sc, acc_sc, output_gf, loss_gf, acc_gf= self.model.build_train_graph(
                                         self.img, self.label_sc, self.label_gf, self.keep_prob)

        self.train_op = train_op
        self.output_sc, self.output_gf = output_sc, output_gf
        self.loss_sc, self.loss_gf = loss_sc, loss_gf
        self.acc_sc, self.acc_gf = acc_sc, acc_gf

        self.train_writer = tf.summary.FileWriter(log_dir + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(log_dir + '/test')
        self.train_writer.add_graph(self.sess.graph)
        
        tf.summary.scalar('Loss_sc', self.loss_sc)
        tf.summary.scalar('Acc_sc', self.acc_sc)
        tf.summary.scalar('Loss_gf', self.loss_gf)
        tf.summary.scalar('Acc_gf', self.acc_gf)

        self.merged = tf.summary.merge_all()

        saver = tf.train.Saver()
        self.saver = saver
        self.sess.run(tf.global_variables_initializer())
        #self.saver.restore(self.sess, './checkpoint/model.ckpt')

    def get_test_data(self):
        test_file_list = os.listdir(self.test_file_path)
        test_file = test_file_list[self.test_index]
        self.test_index += 1
        if self.test_index == len(test_file_list):
            self.test_index = 0
        # read hdf5 test file
        ff = h5py.File(os.path.join(self.test_file_path, test_file), 'r')
        test_img = np.array([ff['img'+str(i)] for i in range(4)]).reshape(-1, 4, 60, 108, 3)
        test_label_gf = np.array(ff['label_gf'][:].reshape(-1, 2))
        test_label_sc = np.array(ff['action_id'][:]).reshape(-1, 1)
        return test_img, test_label_sc, test_label_gf


    def train(self, checkpoint_dir, data_dir):
        assert os.path.isfile(data_dir)

        train_img, train_label, train_label_gf = train_test_data(data_dir, self.params.seq_len)
        test_img, test_label, test_label_gf = train_test_data(self.test_file_path, self.params.seq_len)
        test_img = test_img.astype(np.float32) / 255.
        
        step = 0
        for e in range(self.params.epoch):
            train_loss_sum = 0
            train_acc_sum = 0
            train_gf_loss_sum = 0
            train_gf_acc_sum = 0
            count = 0
            
            for batch_img, batch_label_sc, batch_label_gf in next_batch(train_img, train_label, train_label_gf, self.params.batch_size):
                if batch_img.shape[0] != self.params.batch_size:
                    continue
                batch_img = batch_img.astype(np.float32) / 255.

                _, loss, acc, loss_gf, acc_gf = self.sess.run([self.train_op, self.loss_sc, self.acc_sc, self.loss_gf, self.acc_gf], 
                                              feed_dict={self.img: batch_img,
                                                         self.label_sc: batch_label_sc[:, -1, :].reshape((-1, 1)),
                                                         self.label_gf: batch_label_gf[:, -1, :].reshape((-1, 2)),
                                                         self.keep_prob: 0.5})
                train_loss_sum += loss
                train_acc_sum += acc
                train_gf_loss_sum += loss_gf
                train_gf_acc_sum += acc_gf
                count += 1
                step += 1

                if step % 50 == 0:
                    # Get Train Summary for one batch and add summary to TensorBoard
                    summary = self.sess.run(self.merged, feed_dict={self.img: batch_img, 
                                                                    self.label_sc: batch_label_sc[:, -1, :].reshape((-1, 1)),
                                                                    self.label_gf: batch_label_gf[:, -1, :].reshape((-1, 2)),
                                                                    self.keep_prob: 1})
                    self.train_writer.add_summary(summary, step)
                    self.train_writer.flush()

                    # Get Test Summary on random 32 test images and add summary to TensorBoard
                    test_idx = np.arange(test_img.shape[0])
                    np.random.shuffle(test_idx)
                    test_idx = test_idx[:32]
                    summary = self.sess.run(self.merged, feed_dict={self.img: test_img[test_idx,...], 
                                                                    self.label_sc: test_label[test_idx, ...][:, -1, :].reshape((-1, 1)),
                                                                    self.label_gf: test_label_gf[test_idx, ...][:, -1, :].reshape((-1, 2)),
                                                                    self.keep_prob: 1})
                    self.test_writer.add_summary(summary, step)
                    self.test_writer.flush()

            test_acc_sc_sum = 0
            test_acc_gf_sum = 0
            test_count = 0
            #Feed forward all test images into graph and log accuracy
            for batch_test_img, batch_test_label_sc, batch_test_label_gf in next_batch(test_img, test_label, test_label_gf, 1):
                test_acc_sc, test_acc_gf = self.sess.run([self.acc_sc, self.acc_gf], feed_dict={self.img: batch_test_img, 
                    self.label_sc: batch_test_label_sc[:, -1, :].reshape((-1, 1)),
                    self.label_gf: batch_test_label_gf[:, -1, :].reshape((-1, 2)),
                    self.keep_prob: 1})

                test_acc_sc_sum += test_acc_sc
                test_acc_gf_sum += test_acc_gf
                test_count += 1
            print("epoch %d, loss_sc %.4f, loss_gf %.4f, train_acc_sc %.4f, test_acc_sc %.4f, train_acc_gf %.4f, test_acc_gf %.4f" %
                    (e, train_loss_sum / count, train_gf_loss_sum / count, train_acc_sum / count, test_acc_sc_sum / test_count, train_gf_acc_sum / count,
                        test_acc_gf_sum / test_count))
                
        if checkpoint_dir is not None:
            self.save(checkpoint_dir)
            print("Model has been saved.")

    def save(self, checkpoint_dir):
        ckpt_path = os.path.join(checkpoint_dir, "model.ckpt")
        self.saver.save(self.sess, ckpt_path)

    def load(self, checkpoint_dir):
        ckpt_path = os.path.join(checkpoint_dir, "model.ckpt")
        if os.path.isfile(ckpt_path):
            self.saver.restore(self.sess, ckpt_path)

