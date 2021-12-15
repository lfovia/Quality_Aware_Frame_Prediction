import tensorflow as tf
import numpy as np
from scipy.misc import imsave
from skimage.transform import resize
from copy import deepcopy
import os
from QA_loss_function import Straight_ness_double_derivatives_deriv_gen0
from QA_loss_function import Straight_ness_double_derivatives_deriv_gen1
from QA_loss_function import Straight_ness_double_derivatives_deriv_gen2
from QA_loss_function import Straight_ness_double_derivatives_deriv_gen3
from QA_loss_function import Straight_ness_double_derivatives_deriv_gen4
from QA_loss_function import Straight_ness_double_derivatives_deriv_input_frames
from discriminator2_model import discriminator_2
import constants as c
from loss_functions import combined_loss
from utils import psnr_error, sharp_diff_error
from tfutils import w, b
import pdb
# noinspection PyShadowingNames
class GeneratorModel:
    def __init__(self, session, summary_writer, height_train, width_train, height_test,
                 width_test, scale_layer_fms, scale_kernel_sizes):
        """
        Initializes a GeneratorModel.

        @param session: The TensorFlow Session.
        @param summary_writer: The writer object to record TensorBoard summaries
        @param height_train: The height of the input images for training.
        @param width_train: The width of the input images for training.
        @param height_test: The height of the input images for testing.
        @param width_test: The width of the input images for testing.
        @param scale_layer_fms: The number of feature maps in each layer of each scale network.
        @param scale_kernel_sizes: The size of the kernel for each layer of each scale network.

        @type session: tf.Session
        @type summary_writer: tf.train.SummaryWriter
        @type height_train: int
        @type width_train: int
        @type height_test: int
        @type width_test: int
        @type scale_layer_fms: list<list<int>>
        @type scale_kernel_sizes: list<list<int>>
        """
        self.sess = session
        self.summary_writer = summary_writer
        self.height_train = height_train
        self.width_train = width_train
        self.height_test = height_test
        self.width_test = width_test
        self.scale_layer_fms = scale_layer_fms
        self.scale_kernel_sizes = scale_kernel_sizes
        self.num_scale_nets = len(scale_layer_fms)

        self.define_graph()

    # noinspection PyAttributeOutsideInit
    def define_graph(self):
        """
        Sets up the model graph in TensorFlow.
        """
        with tf.name_scope('generator'):
            ##
            # Data
            ##

            with tf.name_scope('data'):
                self.input_frames_train = tf.placeholder(
                    tf.float32, shape=[None, self.height_train, self.width_train, 3 * c.HIST_LEN])
                self.gt_frames_train = tf.placeholder(
                    tf.float32, shape=[None, self.height_train, self.width_train, 3])

                self.input_frames_test = tf.placeholder(
                    tf.float32, shape=[None, self.height_test, self.width_test, 3 * c.HIST_LEN])
                self.gt_frames_test = tf.placeholder(
                    tf.float32, shape=[None, self.height_test, self.width_test, 3])

                # use variable batch_size for more flexibility
                self.batch_size_train = tf.shape(self.input_frames_train)[0]
                self.batch_size_test = tf.shape(self.input_frames_test)[0]

            ##
            # Scale network setup and calculation
            ##

            self.summaries_train = []
            self.scale_preds_train = []  # the generated images at each scale
            self.scale_gts_train = []  # the ground truth images at each scale
            self.d_scale_preds = []  # the predictions from the discriminator model

            self.summaries_test = []
            self.scale_preds_test = []  # the generated images at each scale
            self.scale_gts_test = []  # the ground truth images at each scale

            for scale_num in xrange(self.num_scale_nets):
                with tf.name_scope('scale_' + str(scale_num)):
                    with tf.name_scope('setup'):
                        ws = []
                        bs = []

                        # create weights for kernels
                        for i in xrange(len(self.scale_kernel_sizes[scale_num])):
                            ws.append(w([self.scale_kernel_sizes[scale_num][i],
                                         self.scale_kernel_sizes[scale_num][i],
                                         self.scale_layer_fms[scale_num][i],
                                         self.scale_layer_fms[scale_num][i + 1]]))
                            bs.append(b([self.scale_layer_fms[scale_num][i + 1]]))

                    with tf.name_scope('calculation'):
                        def calculate(height, width, inputs, gts, last_gen_frames):
                            # scale inputs and gts
                            scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
                            scale_height = int(height * scale_factor)
                            scale_width = int(width * scale_factor)

                            inputs = tf.image.resize_images(inputs, [scale_height, scale_width])
                            scale_gts = tf.image.resize_images(gts, [scale_height, scale_width])

                            # for all scales but the first, add the frame generated by the last
                            # scale to the input
                            if scale_num > 0:
                                last_gen_frames = tf.image.resize_images(
                                    last_gen_frames,[scale_height, scale_width])
                                inputs = tf.concat([inputs, last_gen_frames],axis=3)

                            # generated frame predictions
                            preds = inputs

                            # perform convolutions
                            with tf.name_scope('convolutions'):
                                for i in xrange(len(self.scale_kernel_sizes[scale_num])):
                                    # Convolve layer
                                    preds = tf.nn.conv2d(
                                        preds, ws[i], [1, 1, 1, 1], padding=c.PADDING_G)

                                    # Activate with ReLU (or Tanh for last layer)
                                    if i == len(self.scale_kernel_sizes[scale_num]) - 1:
                                        preds = tf.nn.tanh(preds + bs[i])
                                    else:
                                        preds = tf.nn.relu(preds + bs[i])

                            return preds, scale_gts

                        ##
                        # Perform train calculation
                        ##

                        # for all scales but the first, add the frame generated by the last
                        # scale to the input
                        if scale_num > 0:
                            last_scale_pred_train = self.scale_preds_train[scale_num - 1]
                        else:
                            last_scale_pred_train = None

                        # calculate
                        train_preds, train_gts = calculate(self.height_train,
                                                           self.width_train,
                                                           self.input_frames_train,
                                                           self.gt_frames_train,
                                                           last_scale_pred_train)
                        self.scale_preds_train.append(train_preds)
                        self.scale_gts_train.append(train_gts)

                        # We need to run the network first to get generated frames, run the
                        # discriminator on those frames to get d_scale_preds, then run this
                        # again for the loss optimization.
                        if c.ADVERSARIAL:
                            self.d_scale_preds.append(tf.placeholder(tf.float32, [None, 1]))

                        ##
                        # Perform test calculation
                        ##

                        # for all scales but the first, add the frame generated by the last
                        # scale to the input
                        if scale_num > 0:
                            last_scale_pred_test = self.scale_preds_test[scale_num - 1]
                        else:
                            last_scale_pred_test = None

                        # calculate
                        test_preds, test_gts = calculate(self.height_test,
                                                         self.width_test,
                                                         self.input_frames_test,
                                                         self.gt_frames_test,
                                                         last_scale_pred_test)
                        self.scale_preds_test.append(test_preds)
                        self.scale_gts_test.append(test_gts)

            ##
            # Training
            ##

            with tf.name_scope('train'):
                # global loss is the combined loss from every scale network
                self.global_loss = combined_loss(self.scale_preds_train,
                                                 self.scale_gts_train,self.input_frames_train,
                                                 self.d_scale_preds)
                self.global_step = tf.Variable(0, trainable=False)
                
                lgn_gen = Straight_ness_double_derivatives_deriv_gen4(self.scale_preds_train[3],128,8)
                lgn_gen_test = Straight_ness_double_derivatives_deriv_gen4(self.scale_preds_test[3],128,8)
                lgn_gen_gt = Straight_ness_double_derivatives_deriv_gen4(self.scale_gts_train[3],128,8)
                lgn_gen_gt_0 = Straight_ness_double_derivatives_deriv_gen0(self.scale_gts_train[2],128,8)
                output1gt = lgn_gen_gt[0]
                output1gen_test = lgn_gen_test[0]
                output1gen_train = lgn_gen[0]
                #pdb.set_trace()
                prev_train_frame = self.input_frames_train[:,:,:,9:12]
                prev_train_test = self.input_frames_test[:,:,:,9:12]
                #pdb.set_trace()
                self.lgn_train = Straight_ness_double_derivatives_deriv_gen4(prev_train_frame,128,8)
                self.lgn_test = Straight_ness_double_derivatives_deriv_gen4(prev_train_test,128,8)
                ##################
                output1gt1 = self.lgn_train[0]
                output1gen_test1 = self.lgn_test[0]
                output1gen_train1 = self.lgn_train[0]
                ##################
                ######################
                prev_prev_train_frame = self.input_frames_train[:,:,:,6:9]
                prev_prev_train_test = self.input_frames_test[:,:,:,6:9]
                self.lgn_trainprev = Straight_ness_double_derivatives_deriv_gen4(prev_prev_train_frame,128,8)
                self.lgn_testprev = Straight_ness_double_derivatives_deriv_gen4(prev_prev_train_test,128,8)
                output1gt2 = self.lgn_trainprev[0]
                output1gen_test2 = self.lgn_testprev[0]
                output1gen_train2 = self.lgn_trainprev[0]
                #pdb.set_trace()
                self.lgn_trainprev = Straight_ness_double_derivatives_deriv_gen0(prev_prev_train_frame,128,8)
                self.lgn_testprev = Straight_ness_double_derivatives_deriv_gen0(prev_prev_train_test,128,8)
                ######### derivative 1 ###################
                first = self.lgn_trainprev[0] - self.lgn_train[0]
                ########## derivative 2 ###################
                second = self.lgn_train[0] - lgn_gen_gt[0]
                ########## double derivative ###############
                self.real_double_derivative = tf.reduce_mean(first - second)
                self.meanr, self.variancer = tf.nn.moments(first - second, [0])
                #################################################
                firstgen = self.lgn_trainprev[0] - self.lgn_train[0]
                secondgen = self.lgn_train[0] - lgn_gen[0]
                self.fake_double_derivative = tf.reduce_mean(firstgen - secondgen)
                self.meanf, self.variancef = tf.nn.moments(firstgen - secondgen, [0])
                ################ the same loss at all levels #############################
                self.lgn_train = Straight_ness_double_derivatives_deriv_gen1(prev_train_frame,128,8)
                self.lgn_test = Straight_ness_double_derivatives_deriv_gen1(prev_train_test,128,8)
                lgn_gen_gt = Straight_ness_double_derivatives_deriv_gen1(self.scale_gts_train[3],128,8)
                prev_prev_train_frame = self.input_frames_train[:,:,:,6:9]
                prev_prev_train_test = self.input_frames_test[:,:,:,6:9]
                #pdb.set_trace()
                self.lgn_trainprev = Straight_ness_double_derivatives_deriv_gen1(prev_prev_train_frame,128,8)
                self.lgn_testprev = Straight_ness_double_derivatives_deriv_gen1(prev_prev_train_test,128,8)
                lgn_gen = Straight_ness_double_derivatives_deriv_gen1(self.scale_preds_train[3],128,8)
                ######### derivative 1 ###################
                first = self.lgn_trainprev[0] - self.lgn_train[0]
                ########## derivative 2 ###################
                second = self.lgn_train[0] - lgn_gen_gt[0]
                ########## double derivative ###############
                self.real_double_derivative = tf.reduce_mean(first - second)
                self.meanr, self.variancer = tf.nn.moments(first - second, [0])
                #################################################
                firstgen = self.lgn_trainprev[0] - self.lgn_train[0]
                secondgen = self.lgn_train[0] - lgn_gen[0]
                self.fake_double_derivative = tf.reduce_mean(firstgen - secondgen)
                self.meanf1, self.variancef1 = tf.nn.moments(firstgen - secondgen, [0])
                ######################################### level2 ################################
                self.lgn_train = Straight_ness_double_derivatives_deriv_gen2(prev_train_frame,128,8)
                self.lgn_test = Straight_ness_double_derivatives_deriv_gen2(prev_train_test,128,8)
                lgn_gen_gt = Straight_ness_double_derivatives_deriv_gen2(self.scale_gts_train[3],128,8)
                prev_prev_train_frame = self.input_frames_train[:,:,:,6:9]
                prev_prev_train_test = self.input_frames_test[:,:,:,6:9]
                #pdb.set_trace()
                self.lgn_trainprev = Straight_ness_double_derivatives_deriv_gen2(prev_prev_train_frame,128,8)
                self.lgn_testprev = Straight_ness_double_derivatives_deriv_gen2(prev_prev_train_test,128,8)
                lgn_gen = Straight_ness_double_derivatives_deriv_gen2(self.scale_preds_train[3],128,8)
                ######### derivative 1 ###################
                first = self.lgn_trainprev[0] - self.lgn_train[0]
                ########## derivative 2 ###################
                second = self.lgn_train[0] - lgn_gen_gt[0]
                ########## double derivative ###############
                self.real_double_derivative = tf.reduce_mean(first - second)
                self.meanr, self.variancer = tf.nn.moments(first - second, [0])
                #################################################
                firstgen = self.lgn_trainprev[0] - self.lgn_train[0]
                secondgen = self.lgn_train[0] - lgn_gen[0]
                self.fake_double_derivative = tf.reduce_mean(firstgen - secondgen)
                self.meanf2, self.variancef2 = tf.nn.moments(firstgen - secondgen, [0])
                ######################################### level3 ######################################
                self.lgn_train = Straight_ness_double_derivatives_deriv_gen3(prev_train_frame,128,8)
                self.lgn_test = Straight_ness_double_derivatives_deriv_gen3(prev_train_test,128,8)
                lgn_gen = Straight_ness_double_derivatives_deriv_gen3(self.scale_preds_train[3],128,8)
                lgn_gen_gt = Straight_ness_double_derivatives_deriv_gen3(self.scale_gts_train[3],128,8)
                prev_prev_train_frame = self.input_frames_train[:,:,:,6:9]
                prev_prev_train_test = self.input_frames_test[:,:,:,6:9]
                #pdb.set_trace()
                self.lgn_trainprev = Straight_ness_double_derivatives_deriv_gen3(prev_prev_train_frame,128,8)
                self.lgn_testprev = Straight_ness_double_derivatives_deriv_gen3(prev_prev_train_test,128,8)
                ######### derivative 1 ###################
                first = self.lgn_trainprev[0] - self.lgn_train[0]
                ########## derivative 2 ###################
                second = self.lgn_train[0] - lgn_gen_gt[0]
                ########## double derivative ###############
                self.real_double_derivative = tf.reduce_mean(first - second)
                self.meanr3, self.variancer3 = tf.nn.moments(first - second, [0])
                #################################################
                firstgen = self.lgn_trainprev[0] - self.lgn_train[0]
                secondgen = self.lgn_train[0] - lgn_gen[0]
                self.fake_double_derivative = tf.reduce_mean(firstgen - secondgen)
                self.meanf3, self.variancef3= tf.nn.moments(firstgen - secondgen, [0])
                ############################################### level4 ################################
                self.lgn_train = Straight_ness_double_derivatives_deriv_gen4(prev_train_frame,128,8)
                self.lgn_test = Straight_ness_double_derivatives_deriv_gen4(prev_train_test,128,8)
                lgn_gen = Straight_ness_double_derivatives_deriv_gen4(self.scale_preds_train[3],128,8)
                lgn_gen_gt = Straight_ness_double_derivatives_deriv_gen4(self.scale_gts_train[3],128,8)
                prev_prev_train_frame = self.input_frames_train[:,:,:,6:9]
                prev_prev_train_test = self.input_frames_test[:,:,:,6:9]
                #pdb.set_trace()
                self.lgn_trainprev = Straight_ness_double_derivatives_deriv_gen4(prev_prev_train_frame,128,8)
                self.lgn_testprev = Straight_ness_double_derivatives_deriv_gen4(prev_prev_train_test,128,8)
                ######### derivative 1 ###################
                first = self.lgn_trainprev[0] - self.lgn_train[0]
                ########## derivative 2 ###################
                second = self.lgn_train[0] - lgn_gen_gt[0]
                ########## double derivative ###############
                self.real_double_derivative = tf.reduce_mean(first - second)
                self.meanr, self.variancer = tf.nn.moments(first - second, [0])
                #################################################
                firstgen = self.lgn_trainprev[0] - self.lgn_train[0]
                secondgen = self.lgn_train[0] - lgn_gen[0]
                self.fake_double_derivative = tf.reduce_mean(firstgen - secondgen)
                self.meanf4, self.variancef4 = tf.nn.moments(firstgen - secondgen, [0])
                ############################################################################################
                #################################################
                self.in_1 = tf.stack([output1gen_train],3)[:,:,:,:,0]
                self.in_2 = tf.stack([output1gt],3)[:,:,:,:,0]
                self.in_test = tf.stack([output1gen_test],3)[:,:,:,:,0]
                pdb.set_trace()
                self.discrimator_lossgt = discriminator_2(self.in_2,reuse=False)
                self.discrimator_lossgt_test = discriminator_2(self.in_test,reuse=True)
                self.discrimator_lossgt_train = discriminator_2(self.in_1,reuse=True)
                #self.discrimator_loss = discriminator_2(output1,reuse=True)
                self.loss_d = tf.reduce_mean(self.discrimator_lossgt) - tf.reduce_mean(self.discrimator_lossgt_train)
                ##################
                self.loss_g = tf.reduce_mean(self.discrimator_lossgt_test)
                ############ truth cosine distance #################
                ################### generated cosine distance ###########
                normalize_a = tf.nn.l2_normalize(self.in_1,0)        
                normalize_b = tf.nn.l2_normalize(self.lgn_train,0)[0]
                normalize_aa = tf.nn.l2_normalize(self.in_2,0) 
                #pdb.set_trace()
                
                ############  double derivative loss #################
                
                #self.gen_cos_similarity = tf.reduce_sum(tf.multiply(normalize_a,normalize_b))
                #self.real_cos_similarity = tf.reduce_sum(tf.multiply(normalize_aa,normalize_b))
                #pdb.set_trace()
                optimizer2 = tf.train.AdamOptimizer(learning_rate=2e-8, beta1=0.1, beta2=0.9)
                
                update_ops1 = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator2')
                d_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator2')
                with tf.control_dependencies(update_ops1):
                    self.d_train2 = optimizer2.minimize(self.loss_d, var_list=d_vars1)
                self.global_loss2 = self.global_loss + tf.reduce_mean(self.meanf) + tf.reduce_mean(self.meanf1) +  tf.reduce_mean(self.meanf2) +  tf.reduce_mean(self.meanf3) + tf.reduce_mean(self.meanf4) 
                self.optimizer = tf.train.AdamOptimizer(learning_rate=c.LRATE_G, name='optimizer')
                self.train_op = self.optimizer.minimize(self.global_loss2,
                                                        global_step=self.global_step,
                                                        name='train_op')
                # train loss summary
                loss_summary = tf.summary.scalar('train_loss_G', self.global_loss)
                loss_summary2 = tf.summary.scalar('train_loss_G2', self.global_loss2)
                loss_summary3 = tf.summary.scalar('disc_lgn_loss', self.loss_d)
                loss_summary4 = tf.summary.scalar('g_loss', self.loss_g)
                
                loss_summary42_doublederivative = tf.summary.scalar('fake_doublederivative', self.fake_double_derivative)
                loss_summary42_doublederivative2 = tf.summary.scalar('real_doublederivative', self.real_double_derivative)
                meanr = tf.summary.scalar('real_doublederivative_mean', tf.reduce_mean(self.meanr))
                meanf = tf.summary.scalar('fake_doublederivative_mean', tf.reduce_mean(self.meanf))
                varf = tf.summary.scalar('fake_doublederivative_var', tf.reduce_mean(self.variancef))
                varr = tf.summary.scalar('real_doublederivative_var', tf.reduce_mean(self.variancer))
                self.summaries_train.append(loss_summary)
                self.summaries_train.append(loss_summary2)
                self.summaries_train.append(loss_summary3)
                self.summaries_train.append(loss_summary4)
                
                self.summaries_train.append(loss_summary42_doublederivative)
                self.summaries_train.append(loss_summary42_doublederivative2)
                self.summaries_train.append(meanr)
                self.summaries_train.append(meanf)
                self.summaries_train.append(varf)
                self.summaries_train.append(varr)
                

            ##
            # Error
            ##

            with tf.name_scope('error'):
                # error computation
                # get error at largest scale
                self.psnr_error_train = psnr_error(self.scale_preds_train[-1],
                                                   self.gt_frames_train)
                self.sharpdiff_error_train = sharp_diff_error(self.scale_preds_train[-1],
                                                              self.gt_frames_train)
                self.psnr_error_test = psnr_error(self.scale_preds_test[-1],
                                                  self.gt_frames_test)
                self.sharpdiff_error_test = sharp_diff_error(self.scale_preds_test[-1],
                                                             self.gt_frames_test)
                # train error summaries
                summary_psnr_train = tf.summary.scalar('train_PSNR',
                                                       self.psnr_error_train)
                summary_sharpdiff_train = tf.summary.scalar('train_SharpDiff',
                                                            self.sharpdiff_error_train)
                self.summaries_train += [summary_psnr_train, summary_sharpdiff_train]

                # test error
                summary_psnr_test = tf.summary.scalar('test_PSNR',
                                                      self.psnr_error_test)
                summary_sharpdiff_test = tf.summary.scalar('test_SharpDiff',
                                                           self.sharpdiff_error_test)
                self.summaries_test += [summary_psnr_test, summary_sharpdiff_test]

            # add summaries to visualize in TensorBoard
            self.summaries_train = tf.summary.merge(self.summaries_train)
            self.summaries_test = tf.summary.merge(self.summaries_test)

    def train_step(self, batch,batch2, discriminator=None):
        """
        Runs a training step using the global loss on each of the scale networks.

        @param batch: An array of shape
                      [c.BATCH_SIZE x self.height x self.width x (3 * (c.HIST_LEN + 1))].
                      The input and output frames, concatenated along the channel axis (index 3).
        @param discriminator: The discriminator model. Default = None, if not adversarial.

        @return: The global step.
        """
        ##
        # Split into inputs and outputs
        ##

        input_frames = batch[:, :, :, :-3]
        gt_frames = batch[:, :, :, -3:]
        input_frames2 = batch2[:, :, :, :-3]
        gt_frames2 = batch2[:, :, :, -3:]
        
        ##
        # Train
        ##
        #pdb.set_trace()
        feed_dict = {self.input_frames_train: input_frames, self.gt_frames_train: gt_frames,self.input_frames_test: input_frames2, self.gt_frames_test: gt_frames}

        if c.ADVERSARIAL:
            # Run the generator first to get generated frames
            scale_preds = self.sess.run(self.scale_preds_train, feed_dict=feed_dict)

            # Run the discriminator nets on those frames to get predictions
            d_feed_dict = {}
            for scale_num, gen_frames in enumerate(scale_preds):
                d_feed_dict[discriminator.scale_nets[scale_num].input_frames] = gen_frames
            d_scale_preds = self.sess.run(discriminator.scale_preds, feed_dict=d_feed_dict)

            # Add discriminator predictions to the
            for i, preds in enumerate(d_scale_preds):
                feed_dict[self.d_scale_preds[i]] = preds

        _,_, global_loss, global_psnr_error, global_sharpdiff_error, global_step, summaries,ii1,ii2,prev = \
            self.sess.run([self.train_op,self.d_train2,
                           self.global_loss,
                           self.psnr_error_train,
                           self.sharpdiff_error_train,
                           self.global_step,
                           self.summaries_train,self.in_1,self.in_2,self.lgn_train],
                          feed_dict=feed_dict)
        
        ##
        # User output
        ##
        #pdb.set_trace()
        if global_step % c.STATS_FREQ == 0:
            print 'GeneratorModel : Step ', global_step
            print '                 Global Loss    : ', global_loss
            print '                 PSNR Error     : ', global_psnr_error
            print '                 Sharpdiff Error: ', global_sharpdiff_error
        if global_step % c.SUMMARY_FREQ == 0:
            self.summary_writer.add_summary(summaries, global_step)
            print("pariyesh")
            #pdb.set_trace()
            print 'GeneratorModel: saved summaries'
        if global_step % c.IMG_SAVE_FREQ == 0:
            print '-' * 30
            print 'Saving images...'

            # if not adversarial, we didn't get the preds for each scale net before for the
            # discriminator prediction, so do it now
            if not c.ADVERSARIAL:
                scale_preds = self.sess.run(self.scale_preds_train, feed_dict=feed_dict)

            # re-generate scale gt_frames to avoid having to run through TensorFlow.
            scale_gts = []
            for scale_num in xrange(self.num_scale_nets):
                scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
                scale_height = int(self.height_train * scale_factor)
                scale_width = int(self.width_train * scale_factor)

                # resize gt_output_frames for scale and append to scale_gts_train
                scaled_gt_frames = np.empty([c.BATCH_SIZE, scale_height, scale_width, 3])
                for i, img in enumerate(gt_frames):
                    # for skimage.transform.resize, images need to be in range [0, 1], so normalize
                    # to [0, 1] before resize and back to [-1, 1] after
                    sknorm_img = (img / 2) + 0.5
                    resized_frame = resize(sknorm_img, [scale_height, scale_width, 3])
                    scaled_gt_frames[i] = (resized_frame - 0.5) * 2
                scale_gts.append(scaled_gt_frames)

            # for every clip in the batch, save the inputs, scale preds and scale gts
            for pred_num in xrange(len(input_frames)):
                pred_dir = c.get_dir(os.path.join(c.IMG_SAVE_DIR, 'Step_' + str(global_step),
                                                  str(pred_num)))

                # save input images
                for frame_num in xrange(c.HIST_LEN):
                    img = input_frames[pred_num, :, :, (frame_num * 3):((frame_num + 1) * 3)]
                    imsave(os.path.join(pred_dir, 'input_' + str(frame_num) + '.png'), img)

                # save preds and gts at each scale
                # noinspection PyUnboundLocalVariable
                for scale_num, scale_pred in enumerate(scale_preds):
                    gen_img = scale_pred[pred_num]

                    path = os.path.join(pred_dir, 'scale' + str(scale_num))
                    gt_img = scale_gts[scale_num][pred_num]

                    imsave(path + '_gen.png', gen_img)
                    imsave(path + '_gt.png', gt_img)

            print 'Saved images!'
            print '-' * 30

        return global_step

    def test_batch(self, batch, global_step, num_rec_out=1, save_imgs=True):
        """
        Runs a training step using the global loss on each of the scale networks.

        @param batch: An array of shape
                      [batch_size x self.height x self.width x (3 * (c.HIST_LEN+ num_rec_out))].
                      A batch of the input and output frames, concatenated along the channel axis
                      (index 3).
        @param global_step: The global step.
        @param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
                            using previously-generated frames as input. Default = 1.
        @param save_imgs: Whether or not to save the input/output images to file. Default = True.

        @return: A tuple of (psnr error, sharpdiff error) for the batch.
        """
        if num_rec_out < 1:
            raise ValueError('num_rec_out must be >= 1')

        print '-' * 30
        print 'Testing:'

        ##
        # Split into inputs and outputs
        ##

        input_frames = batch[:, :, :, :3 * c.HIST_LEN]
        gt_frames = batch[:, :, :, 3 * c.HIST_LEN:]
        #pdb.set_trace()
        ##
        # Generate num_rec_out recursive predictions
        ##

        working_input_frames = deepcopy(input_frames)  # input frames that will shift w/ recursion
        rec_preds = []
        rec_summaries = []
        for rec_num in xrange(num_rec_out):
            working_gt_frames = gt_frames[:, :, :, 3 * rec_num:3 * (rec_num + 1)]

            feed_dict = {self.input_frames_test: working_input_frames,
                         self.gt_frames_test: working_gt_frames}
            preds, psnr, sharpdiff, summaries = self.sess.run([self.scale_preds_test[-1],
                                                               self.psnr_error_test,
                                                               self.sharpdiff_error_test,
                                                               self.summaries_test],
                                                              feed_dict=feed_dict)

            # remove first input and add new pred as last input
            working_input_frames = np.concatenate(
                [working_input_frames[:, :, :, 3:], preds], axis=3)

            # add predictions and summaries
            rec_preds.append(preds)
            rec_summaries.append(summaries)

            print 'Recursion ', rec_num
            print 'PSNR Error     : ', psnr
            print 'Sharpdiff Error: ', sharpdiff

        # write summaries
        # TODO: Think of a good way to write rec output summaries - rn, just using first output.
        self.summary_writer.add_summary(rec_summaries[0], global_step)

        ##
        # Save images
        ##

        if save_imgs:
            for pred_num in xrange(len(input_frames)):
                pred_dir = c.get_dir(os.path.join(
                    c.IMG_SAVE_DIR, 'Tests/Step_' + str(global_step), str(pred_num)))

                # save input images
                for frame_num in xrange(c.HIST_LEN):
                    img = input_frames[pred_num, :, :, (frame_num * 3):((frame_num + 1) * 3)]
                    imsave(os.path.join(pred_dir, 'input_' + str(frame_num) + '.png'), img)

                # save recursive outputs
                for rec_num in xrange(num_rec_out):
                    gen_img = rec_preds[rec_num][pred_num]
                    gt_img = gt_frames[pred_num, :, :, 3 * rec_num:3 * (rec_num + 1)]
                    imsave(os.path.join(pred_dir, 'gen_' + str(rec_num) + '.png'), gen_img)
                    imsave(os.path.join(pred_dir, 'gt_' + str(rec_num) + '.png'), gt_img)

        print '-' * 30
