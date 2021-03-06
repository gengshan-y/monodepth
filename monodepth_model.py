# Copyright UCL Business plc 2017. Patent Pending. All rights reserved. 
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com

"""Fully convolutional model for monocular depth estimation
    by Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
    http://visual.cs.ucl.ac.uk/pubs/monoDepth/
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from bilinear_sampler import *

import pdb

monodepth_parameters = namedtuple('parameters', 
                        'encoder, '
                        'height, width, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'do_stereo, '
                        'use_lidar, '
                        'lidar_name, '
                        'wrap_mode, '
                        'use_deconv, '
                        'use_upproj, '
                        'alpha_image_loss, '
                        'disp_gradient_loss_weight, '
                        'lr_loss_weight, '
                        'from_disp, '
                        'full_summary')

class MonodepthModel(object):
    """monodepth model"""

    def __init__(self, params, mode, left, right, left_lidar=None, right_lidar=None, reuse_variables=None, model_index=0):
        if len(params.lidar_name.split('_')) > 1:
            self.n_class = int(params.lidar_name.split('_')[-1])
        else:
            self.n_class = 0
        # self.step = tf.placeholder(tf.float32, shape=(),name='step')
        self.params = params
        self.mode = mode
        self.left = left
        self.right = right
        if params.use_lidar:
            self.lidar = tf.stack([left_lidar,  right_lidar],  0)
            self.lidar = tf.squeeze(tf.stack([left_lidar,  right_lidar],  -2))
        self.model_collection = ['model_' + str(model_index)]

        self.reuse_variables = reuse_variables

        self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def scale_pyramid_lidar(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_images(img,  [nh,nw], tf.image.ResizeMethod.BILINEAR))
        return scaled_imgs

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)


    def get_smoothness(self, d, img):
        disp_gradients_x = tf.abs(self.gradient_x(d))
        disp_gradients_y = tf.abs(self.gradient_y(d))

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y
        return [smoothness_x,  smoothness_y]

    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    def get_disp(self, x):
        # disp = 0.3 * self.conv(x, 2, 3, 1, tf.nn.sigmoid)
        # disp = 0.3 * self.conv(x, 20, 3, 1, tf.nn.sigmoid)
        disp = 0.3 * self.conv(x, 1, 3, 1, tf.nn.sigmoid)
        return disp

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn,normalizer_fn = slim.batch_norm)
        # return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x,         num_layers, 1, 1)
        conv2 = self.conv(conv1,     num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:,3:-1,3:-1,:]

    def upproj(self, x, num_out_layers, scale):
        upsample = self.upsample_nn(x, scale)
        conv1_1 = self.conv(upsample, num_out_layers, 5, 1)
        conv1_2 = self.conv(conv1_1, num_out_layers, 3, 1, activation_fn=None)
        conv2 = self.conv(upsample, num_out_layers, 5, 1, activation_fn=None)
        conv = tf.nn.elu(conv1_2 + conv2)
        
        return conv


    def build_resnet50(self):
        #set convenience functions
        conv   = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = conv(self.model_input, 64, 7, 2) # H/2  -   64D
            pool1 = self.maxpool(conv1,           3) # H/4  -   64D
            conv2 = self.resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = self.resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = self.resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = self.resblock(conv4,     512, 3) # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
        
        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv(conv5,   512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,   512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,   256, 3, 1)

            upconv4 = upconv(iconv5,  128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4  = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv(concat3,    64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3  = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv(concat2,    32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2  = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv(concat1,   16, 3, 1)

            if self.n_class:
                self.resized_disp = 0.3 * self.conv(iconv1, self.n_class, 3, 1, None)
                # self.resized_disp = tf.image.resize_images(self.resized_disp, [self.params.height, self.params.width])
            # regression
            else:
                self.disp1 = self.get_disp(iconv1)
                self.resized_disp = self.disp1[:,:,:,:1]
                # self.resized_disp = tf.image.resize_images(self.disp1[:,:,:,:1], [375, 1242])
                # self.resized_disp = self.disp1[:,:,:,:1]

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):

                self.left_pyramid  = self.scale_pyramid(self.left,  4)
                #if self.params.use_lidar:
                #    self.lidar = self.scale_pyramid_lidar(self.lidar,  4);
                if self.mode == 'train':
                    self.right_pyramid = self.scale_pyramid(self.right, 4)
                    # self.resized_im_r = tf.image.resize_images(self.right,[375,1242])

                if self.params.do_stereo:
                    self.model_input = tf.concat([self.left, self.right], 3)
                else:
                    self.model_input = self.left
                # self.resized_im = tf.image.resize_images(self.left,[215,1137])
                # self.resized_im_l = tf.image.resize_images(self.left,[375,1242])
                

                #build model
                if self.params.encoder == 'vgg':
                    self.build_vgg()
                elif self.params.encoder == 'resnet50':
                    self.build_resnet50()
                else:
                    return None

    def build_outputs(self):
        # STORE DISPARITIES
        with tf.variable_scope('disparities'):
            # self.disp_est  = [self.disp1, self.disp2, self.disp3, self.disp4]
            # self.disp_left_est  = [tf.expand_dims(d[:,:,:,0], 3) for d in self.disp_est]
            # self.disp_right_est = [tf.expand_dims(d[:,:,:,1], 3) for d in self.disp_est]
                
            self.disc_pred_l = tf.nn.softmax(self.resized_disp[:,:,:,0:self.n_class])
            self.disc_pred_l = tf.argmax(self.disc_pred_l,axis=-1) 
            self.disc_pred_l = tf.cast( tf.expand_dims(self.disc_pred_l,axis=-1), tf.uint8)

        if self.mode == 'test':
            return

        # GENERATE IMAGES
        # with tf.variable_scope('images'):
            
            # self.left_est  = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i])  for i in range(4)]
            # self.image_mask = [self.left_est[i]>0 for i in range(4)]
            # self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]

        # LR CONSISTENCY
        # with tf.variable_scope('left-right'):
        #     self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i])  for i in range(4)]
        #     self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in range(4)]

        # DISPARITY SMOOTHNESS
        #with tf.variable_scope('smoothness'):
            # self.disp_left_smoothness  = self.get_disparity_smoothness(self.disp_left_est,  self.left_pyramid)
            # self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)
            # self.depth_smoothness = self.get_smoothness(self.resized_disp[:,:,:,0:1], self.resized_im_l) + \
            #                         self.get_smoothness(self.resized_disp[:,:,:,1:2], self.resized_im_r)

        # SUPERVISED
        if self.params.use_lidar:
            with tf.variable_scope('supervised'):
                # upsampling 
                if not self.n_class:
                    self.lidar = self.lidar[:,:,:,:1]
                    self.mask = tf.cast(self.lidar > 0, tf.int32)
                
                    self.est_pure = tf.dynamic_partition(self.resized_disp, self.mask,2)[1]
                    self.gt_pure = tf.dynamic_partition(self.lidar, self.mask,2)[1]

                # log
                #self.mask = [tf.logical_and(self.lidar[i]>0,self.disp_left_est[i]>0) for i in range(4)]
                #self.est_masked = [tf.where(self.mask[i],tf.log(self.disp_left_est[i]),tf.zeros_like(self.lidar[i])) for i in range(4)]
                #self.gt_masked = [tf.where(self.mask[i],tf.log(self.lidar[i]),tf.zeros_like(self.lidar[i])) for i in range(4)]
                
                # linear
                #self.mask = [self.lidar[i]>0 for i in range(4)]
                #self.est_masked = [tf.where(self.mask[i],self.disp_left_est[i],tf.zeros_like(self.lidar[i])) for i in range(4)]
                #self.gt_masked = [tf.where(self.mask[i],self.lidar[i],tf.zeros_like(self.lidar[i])) for i in range(4)]

                #self.mask = [tf.sign(x) for x in self.lidar]
                #self.mask = [tf.cast(x > 0, tf.int32) for x in self.lidar]
                #self.est_masked = [tf.log(tf.dynamic_partition(x,self.mask[i],2)[1]) for i,x in enumerate(self.disp_left_est)]
                #self.gt_masked = [tf.log(tf.dynamic_partition(x,self.mask[i],2)[1]) for i,x in enumerate(self.lidar)]
                # self.mask = [tf.cast(x > 1e-3, tf.float32) for x in self.lidar]
                # self.mask = [tf.sign(x) for x in self.lidar]
                #self.est_masked = [self.mask[i] * tf.log(self.disp_left_est[i]) for i in range(4)]
                #self.gt_masked = [self.mask[i] * tf.log(self.lidar[i]) for i in range(4)]
                #self.est_masked = [tf.where(~tf.is_finite(x), tf.zeros_like(x), x) for  x in self.est_masked]
                #self.gt_masked = [tf.where(~tf.is_finite(x), tf.zeros_like(x), x) for  x in self.gt_masked]

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
       #     # IMAGE RECONSTRUCTION
       #     # L1
       #     self.l1_left = [tf.abs( self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
       #     # self.l1_left = [tf.where(self.image_mask[i], self.l1_left[i], tf.zeros_like(self.l1_left[i])) for i in range(4)]  # mask out boarder
       #     self.l1_reconstruction_loss_left  = [tf.reduce_mean(l) for l in self.l1_left]
       #     self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
       #     self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

       #     # SSIM
       #     self.ssim_left = [self.SSIM( self.left_est[i],  self.left_pyramid[i]) for i in range(4)]
       #     self.ssim_loss_left  = [tf.reduce_mean(s) for s in self.ssim_left]
       #     self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
       #     self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

       #     # WEIGTHED SUM
       #     self.image_loss_right = [self.params.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_right[i] for i in range(4)]
       #     self.image_loss_left  = [self.params.alpha_image_loss * self.ssim_loss_left[i]  + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_left[i]  for i in range(4)]
       #     self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # DISPARITY SMOOTHNESS
            #self.disp_left_loss  = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i]))  / 2 ** i for i in range(4)]
            #self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
            #self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)
            # self.d_gradient_loss = tf.add_n([tf.reduce_mean(x) for x in self.depth_smoothness])

            # LR CONSISTENCY
            #self.lr_left_loss  = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i]))  for i in range(4)]
            #self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in range(4)]
            #self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)

            # TOTAL LOSS
            # self.total_loss =  self.image_loss  + self.params.disp_gradient_loss_weight * self.disp_gradient_loss
            #self.total_loss = self.params.disp_gradient_loss_weight * self.disp_gradient_loss\
            #                  + self.params.lr_loss_weight * self.lr_loss

            # Supervised
            if self.params.use_lidar:
                if self.n_class:
                    self.lidar = self.lidar[:,:,:,:1]
                    self.lidar = tf.subtract(tf.cast(self.lidar,tf.int32),1)
                    self.mask = tf.cast(self.lidar >= 0, tf.int32)
                    self.gt_pure = tf.dynamic_partition(self.lidar, self.mask,2)[1]   

                    # lab_l = tf.one_hot(self.lidar[:,:,:,0],self.n_class)
                    lab_l = 0.5 * tf.one_hot(self.lidar[:,:,:,0],100) + 0.25 * tf.one_hot(self.lidar[:,:,:,0]-1,100) + 0.25 * tf.one_hot(self.lidar[:,:,:,0]+1,100)
                    self.err_dist = tf.nn.softmax_cross_entropy_with_logits(\
                         logits=self.resized_disp[:,:,:,0:self.n_class],labels = lab_l)
                    self.err_dist = tf.dynamic_partition(self.err_dist, self.mask[:,:,:,0],2)[1]
                else:
                    self.err_dist = tf.abs(self.gt_pure - self.est_pure)

                #berhu_dev = 0.2 * tf.reduce_max(self.err_dist)
                #berhu_mask = self.err_dist > berhu_dev
                #self.err_dist = tf.where(berhu_mask, (tf.square(self.err_dist) + berhu_dev * berhu_dev)/(2*berhu_dev), self.err_dist)
                
                # self.err_image = tf.abs(self.est_masked2 - self.gt_masked2)
                #self.err_dist = 0.5 * tf.square(self.gt_pure - self.est_pure)
                #self.err_image = 0.5 * tf.square(self.est_masked2 - self.gt_masked2)

                self.sup_loss = tf.reduce_mean(self.err_dist)
                # self.sup_loss = tf.reduce_mean(self.err_pure_l) + tf.reduce_mean(self.err_pure_r)

                # self.sup_loss = [0.5 * tf.square(self.est_masked[i] - self.gt_masked[i]) for i in range(4)]
                #self.sup_loss = [tf.abs(self.est_masked[i] - self.gt_masked[i]) for i in range(4)]
                #self.sup_loss = [tf.reduce_mean(x) for x in self.sup_loss]
                #self.sup_loss = tf.add_n(self.sup_loss)
                # self.sup_lambda = tf.exp(-10/(self.step+1.))
                self.total_loss = self.sup_loss
                # self.total_loss = self.sup_lambda * self.total_loss + self.image_loss
                # self.total_loss += self.params.disp_gradient_loss_weight * self.d_gradient_loss
                # self.total_loss += self.params.lr_loss_weight * self.lr_loss

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
         #   for i in range(4):
         #       tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i], collections=self.model_collection)
         #       tf.summary.scalar('l1_loss_' + str(i), self.l1_reconstruction_loss_left[i], collections=self.model_collection)
         #       tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i], collections=self.model_collection)
         #       # tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_left_loss[i], collections=self.model_collection)
         #       # tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i], collections=self.model_collection)
         #       # tf.summary.scalar('l1_loss_' + str(i), self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i], collections=self.model_collection)
         #       # tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i], collections=self.model_collection)
         #       # tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_left_loss[i] + self.disp_right_loss[i], collections=self.model_collection)
         #       # tf.summary.scalar('lr_loss_' + str(i), self.lr_left_loss[i] + self.lr_right_loss[i], collections=self.model_collection)
         #       tf.summary.image('disp_left_est_' + str(i), self.disp_left_est[i], max_outputs=4, collections=self.model_collection)
         #       # tf.summary.image('disp_right_est_' + str(i), self.disp_right_est[i], max_outputs=4, collections=self.model_collection)

         #       if self.params.full_summary:
         #           tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4, collections=self.model_collection)
         #           #tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4, collections=self.model_collection)
         #           tf.summary.image('ssim_left_'  + str(i), self.ssim_left[i],  max_outputs=4, collections=self.model_collection)
         #           #tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=4, collections=self.model_collection)
         #           tf.summary.image('l1_left_'  + str(i), self.l1_left[i],  max_outputs=4, collections=self.model_collection)
         #           #tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=4, collections=self.model_collection)

            if self.params.use_lidar:
                # tf.summary.scalar('d_gradient_loss', self.d_gradient_loss, collections=self.model_collection)
                # tf.summary.scalar('sup_lambda', self.sup_lambda, collections=self.model_collection)
                tf.summary.scalar('sup_loss', self.sup_loss, collections=self.model_collection)
                # tf.summary.scalar('pixel_num', tf.reduce_sum(tf.cast(self.mask[0], tf.float32)),collections=self.model_collection) 
                tf.summary.scalar('pixel_num', tf.reduce_sum(tf.cast(self.mask, tf.float32)),collections=self.model_collection)
                # for debug
                # for i in range(4):
                    #tf.summary.histogram('hist_pred', self.est_masked[i], collections=self.model_collection)
                    #tf.summary.histogram('hist_pred_full', self.disp_left_est[i], collections=self.model_collection)
                    #tf.summary.histogram('hist_gt', self.gt_masked[i], collections=self.model_collection)
                #tf.summary.image('hist_gt_masked', self.gt_masked[0], max_outputs=4, collections=self.model_collection)

                # classification
                tf.summary.image('self.disc_pred_l', self.disc_pred_l, max_outputs=4, collections=self.model_collection)
                tf.summary.histogram('self.disc_pred_l', self.disc_pred_l, collections=self.model_collection)
                #tf.summary.histogram('self.est_pure', self.est_pure, collections=self.model_collection)
                tf.summary.histogram('self.gt_pure', self.gt_pure, collections=self.model_collection)
                tf.summary.histogram('self.err_dist', self.err_dist, collections=self.model_collection)
                # tf.summary.histogram('self.resized_disp', self.resized_disp, collections=self.model_collection)
                
                #tf.summary.histogram('self.err_pure_l', self.err_pure_l, collections=self.model_collection)
                #tf.summary.histogram('self.err_pure_r', self.err_pure_r, collections=self.model_collection)
                # tf.summary.histogram('self.err_dist', self.err_dist, collections=self.model_collection)
               
                # tf.summary.image('err_image', self.err_image[:,:,:,0:1] , max_outputs=4, collections=self.model_collection)
                # tf.summary.image('err_image', self.err_image[:,:,:,1:2] , max_outputs=4, collections=self.model_collection)
                # tf.summary.image('est', self.resized_disp[:,:,:,0:1], max_outputs=4, collections=self.model_collection)
                # tf.summary.image('est', self.resized_disp[:,:,:,1:2], max_outputs=4, collections=self.model_collection)

            if self.params.full_summary:
                tf.summary.image('left',  self.left,   max_outputs=4, collections=self.model_collection)
                # tf.summary.image('right', self.right,  max_outputs=4, collections=self.model_collection)

