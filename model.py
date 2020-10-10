from __future__ import print_function

import os
import time
import random

from PIL import Image
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np

from utils import *

def concat(layers):
    return tf.concat(layers, axis=3)

def DecomNet(input_im, layer_num, channel=64, kernel_size=3):
    input_max = tf.reduce_max(input_im, axis=3, keepdims=True)
    input_im = concat([input_max, input_im])
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        conv = tf.layers.conv2d(input_im, channel, kernel_size * 3,
                padding='same', activation=None, name="shallow_feature_extraction")
        for idx in range(layer_num-1):
            conv = tf.layers.conv2d(conv, channel, kernel_size,
                    padding='same', activation=tf.nn.relu, name='activated_layer_%d' % idx)
        idx += 1
        out = tf.layers.conv2d(conv, channel, kernel_size,
                padding='same', activation=tf.nn.relu, name='activated_layer_%d' % idx)

    I = out[..., :channel//4]
    R = out[..., channel//4:]
    return R, I, conv

def Recon(R, I):
    """ reconstruct image """
    conv = concat([R, I])
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        conv = tf.layers.conv2d(conv, 64, 3,
                activation=tf.nn.relu,
                padding='same', name='recon_layer_1')
        conv = tf.layers.conv2d(conv, 3, 3,
                activation=tf.nn.sigmoid,
                padding='same', name='recon_layer')
        return conv

def RelightNet(R, I, conv, channel=64, kernel_size=3):
    input_im = concat([R, I])
    # input_im = conv
    with tf.variable_scope('RelightNet'):
        conv0 = tf.layers.conv2d(input_im, channel, kernel_size, padding='same', activation=None)
        conv1 = tf.layers.conv2d(conv0, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)

        up1 = tf.image.resize_nearest_neighbor(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]))
        deconv1 = tf.layers.conv2d(up1, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv2
        up2 = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
        deconv2= tf.layers.conv2d(up2, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv1
        up3 = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
        deconv3 = tf.layers.conv2d(up3, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv0

        deconv1_resize = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        deconv2_resize = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])
        feature_fusion = tf.layers.conv2d(feature_gather, channel, 1, padding='same', activation=None)
        output = tf.layers.conv2d(feature_fusion, I.shape[-1], 3, padding='same', activation=None)

    return output


class lowlight_enhance(object):
    def __init__(self, sess):
        self.sess = sess
        self.DecomNet_layer_num = 5 - 2

        # build the model
        self.input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

        R_low, I_low, conv = DecomNet(self.input_low, layer_num=self.DecomNet_layer_num)
        R_high, I_high, _ = DecomNet(self.input_high, layer_num=self.DecomNet_layer_num)

        I_delta = RelightNet(R_low, I_low, conv)

        S_high_high = Recon(R_high, I_high)
        S_high_low = Recon(R_high, I_low)
        S_low_low = Recon(R_low, I_low)
        S_low_high = Recon(R_low, I_high)
        S_low_delta = Recon(R_low, I_delta)

        output_R_high = Recon(R_high, tf.zeros_like(I_high))
        output_I_high = Recon(tf.zeros_like(R_low), I_high)

        self.output_R_low = Recon(R_low, tf.zeros_like(I_low))
        self.output_I_low = Recon(tf.zeros_like(R_low), I_low)
        self.output_I_delta = Recon(tf.zeros_like(R_low), I_delta)
        self.output_S = S_low_delta

        def diff(a, b):
            return tf.reduce_mean(tf.abs(a - b))

        def diff_rgb(img):
            return diff(img[...,0], img[...,1]) + diff(img[...,1], img[...,2])

        # loss
        self.recon_loss_low = diff(S_low_low, self.input_low)
        self.recon_loss_high = diff(S_high_high, self.input_high)
        self.recon_loss_mutal_low = diff(S_high_low, self.input_low)
        self.recon_loss_mutal_high = diff(S_low_high, self.input_high)
        self.equal_R_loss = diff(R_low, R_high)
        self.relight_loss = diff(S_low_delta, self.input_high)

        # self.Ismooth_loss_low = self.smooth(self.output_I_low, self.input_low)
        # self.Ismooth_loss_high = self.smooth(output_I_high, self.input_high)
        # self.Ismooth_loss_delta = self.smooth(self.output_I_delta, self.input_high)

        self.Igray_loss_low = diff_rgb(self.output_I_low)
        self.Igray_loss_high = diff_rgb(output_I_high)
        self.Igray_loss_delta = diff_rgb(self.output_I_delta)

        self.loss_Decom = self.recon_loss_low + self.recon_loss_high + \
            + 0.01 * self.recon_loss_mutal_low + 0.01 * self.recon_loss_mutal_high + \
            + 0.01 * self.equal_R_loss \
            # + 0.1 * self.Ismooth_loss_low + 0.1 * self.Ismooth_loss_high
        self.loss_Relight = self.relight_loss
                # + 0.1 * self.Ismooth_loss_delta \

        # use illumincation map gray constraint
        if 0:
            self.loss_Decom += .1 * (self.Igray_loss_low + self.Igray_loss_high)
            self.loss_Relight += .1 * self.Igray_loss_delta

        var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
        var_Relight = [var for var in tf.trainable_variables() if 'RelightNet' in var.name]

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.train_op_Decom = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer').minimize(
                self.loss_Decom, var_list = var_Decom)
        self.train_op_Relight = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer').minimize(
                self.loss_Relight, var_list = var_Relight)

        self.sess.run(tf.global_variables_initializer())

        self.saver_Decom = tf.train.Saver(var_list = var_Decom)
        self.saver_Relight = tf.train.Saver(var_list = var_Relight)

        print("[*] Initialize model successfully...", )

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
        self.smooth_kernel_y = tf.transpose(self.smooth_kernel_x, [1, 0, 2, 3])

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))

    def ave_gradient(self, input_tensor, direction):
        return tf.layers.average_pooling2d(self.gradient(input_tensor, direction), pool_size=3, strides=1, padding='SAME')

    def smooth(self, input_I, input_R):
        input_I = tf.image.rgb_to_grayscale(input_I)
        input_R = tf.image.rgb_to_grayscale(input_R)
        return tf.reduce_mean(self.gradient(input_I, "x") * tf.exp(-10 * self.ave_gradient(input_R, "x"))) + tf.reduce_mean(self.gradient(input_I, "y") * tf.exp(-10 * self.ave_gradient(input_R, "y")))

    def evaluate(self, epoch_num, eval_low_data,
            sample_dir, train_phase,
            eval_high_data=None,
            ):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
            inputs = {self.input_low: input_low_eval}
            if eval_high_data is not None:
                input_high_eval = np.expand_dims(eval_high_data[idx], axis=0)
                inputs[self.input_high] = input_high_eval

            if train_phase == "Decom":
                outputs = [self.output_R_low, self.output_I_low]
            elif train_phase == "Relight":
                outputs = [self.output_S, self.output_I_delta]

            result_1, result_2 = self.sess.run(outputs, feed_dict=inputs)

            save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1, result_2)

    def train(self, train_low_data, train_high_data, eval_low_data,
            batch_size, patch_size, epoch, lr,
            sample_dir, ckpt_dir, eval_every_epoch,
            train_phase,
            eval_high_data=None,
            ):
        assert len(train_low_data) == len(train_high_data)
        numBatch = len(train_low_data) // int(batch_size)

        # load pretrained model
        if train_phase == "Decom":
            train_op = self.train_op_Decom
            train_loss = self.loss_Decom
            saver = self.saver_Decom
        elif train_phase == "Relight":
            train_op = self.train_op_Relight
            train_loss = self.loss_Relight
            saver = self.saver_Relight

        load_model_status, global_step = self.load(saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0

        for epoch in range(start_epoch, epoch):
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)

                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)

                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(list(tmp))
                        train_low_data, train_high_data  = zip(*tmp)

                # train
                _, loss = self.sess.run([train_op, train_loss],
                        feed_dict={
                            self.input_low: batch_input_low,
                            self.input_high: batch_input_high,
                            self.lr: lr[epoch]
                            }
                        )

                iter_num += 1

            print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                    % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))

            # evalutate the model and save a checkpoint file for it
            if (epoch + 1) % int(eval_every_epoch) == 0:
                if 0:
                    self.evaluate(epoch + 1, eval_low_data,
                        sample_dir=sample_dir, train_phase=train_phase,
                        eval_high_data=eval_high_data,
                        )
                self.save(saver, iter_num, ckpt_dir, "RetinexNet-%s" % train_phase)

        print("[*] Finish training for phase %s." % train_phase)

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir, decom_flag):
        tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status_Decom, _ = self.load(self.saver_Decom, './model/Decom')
        load_model_status_Relight, _ = self.load(self.saver_Relight, './model/Relight')
        if load_model_status_Decom and load_model_status_Relight:
            print("[*] Load weights successfully...")

        print("[*] Testing...")
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)

            if int(decom_flag) == 1:
                [R_low, I_low, I_delta, S] = self.sess.run(
                    [self.output_R_low, self.output_I_low, self.output_I_delta, self.output_S],
                    feed_dict = {self.input_low: input_low_test})
                save_images(os.path.join(save_dir, name + "_R_low." + suffix), R_low)
                save_images(os.path.join(save_dir, name + "_I_low." + suffix), I_low)
                save_images(os.path.join(save_dir, name + "_I_delta." + suffix), I_delta)
            else:
                [I_delta, S] = self.sess.run(
                    [self.output_I_delta, self.output_S],
                    feed_dict = {self.input_low: input_low_test})
            save_images(os.path.join(save_dir, name + "_S."   + suffix), S, scale=1)

