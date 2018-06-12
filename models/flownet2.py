import glob, os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge

class Flownet2:
    def __init__(self, bilinear_warping_module):
        self.weights = dict()

        for key, shape in self.all_variables():
            self.weights[key] = tf.get_variable(key, shape=shape)

        self.bilinear_warping_module = bilinear_warping_module

    def leaky_relu(self, x, s):
        assert s > 0 and s < 1, "Wrong s"
        return tf.maximum(x, s*x)

    def warp(self, x, flow):
        return self.bilinear_warping_module.bilinear_warping(x, tf.stack([flow[:,:,:,1], flow[:,:,:,0]], axis=3))

    # flip true -> [:,:,:,0] y axis downwards
    #              [:,:,:,1] x axis
    #   as in matrix indexing
    #
    # false returns 0->x, 1->y
    def __call__(self, im0, im1, flip=True):
        f = self.get_blobs(im0, im1)['predict_flow_final']
        if flip:
            f = tf.stack([f[:,:,:,1], f[:,:,:,0]], axis=3)
        return f

    def get_optimizer(self, flow, target, learning_rate=1e-4):
        #flow = self.__call__(im0, im1)
        loss = tf.reduce_sum(flow * target) # target holding the gradients!
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.95, beta2=0.99, epsilon=1e-8)
        opt = opt.minimize(loss, var_list=
        #    [v for k,v in self.weights.iteritems() if (k.startswith('net3_') or k.startswith('netsd_') or k.startswith('fuse_'))])
            [v for k,v in self.weights.iteritems() if ((k.startswith('net3_') or k.startswith('netsd_') or k.startswith('fuse_')) and not ('upsample' in k or 'deconv' in k))])
        return opt, loss

    # If I run the network with large images (1024x2048) it crashes due to memory
    # constraints on a 12Gb titan X. 
    # See https://github.com/tensorflow/tensorflow/issues/5816#issuecomment-268710077
    # for a possible explanation. I fix it by adding run_after in the section with 
    # the correlation layer so that 441 large tensors are not allocated at the same time
    def run_after(self, a_tensor, b_tensor):
        """Force a to run after b"""
        ge.reroute.add_control_inputs(a_tensor.op, [b_tensor.op])

    # without epsilon I get nan-errors when I backpropagate
    def l2_norm(self, x):
        return tf.sqrt(tf.maximum(1e-5, tf.reduce_sum(x**2, axis=3, keep_dims=True)))

    def get_blobs(self, im0, im1):
        blobs = dict()

        batch_size = tf.to_int32(tf.shape(im0)[0])
        width = tf.to_int32(tf.shape(im0)[2])
        height = tf.to_int32(tf.shape(im0)[1])
        TARGET_WIDTH = width
        TARGET_HEIGHT = height

        divisor = 64.
        ADAPTED_WIDTH = tf.to_int32(tf.ceil(tf.to_float(width)/divisor) * divisor)
        ADAPTED_HEIGHT = tf.to_int32(tf.ceil(tf.to_float(height)/divisor) * divisor)

        SCALE_WIDTH = tf.to_float(width) / tf.to_float(ADAPTED_WIDTH);
        SCALE_HEIGHT = tf.to_float(height) / tf.to_float(ADAPTED_HEIGHT);

        blobs['img0'] = im0
        blobs['img1'] = im1

        blobs['img0s'] = blobs['img0']*0.00392156862745098
        blobs['img1s'] = blobs['img1']*0.00392156862745098

        #mean = np.array([0.411451, 0.432060, 0.450141])
        mean = np.array([0.37655231, 0.39534855, 0.40119368])
        blobs['img0_nomean'] = blobs['img0s'] - mean
        blobs['img1_nomean'] = blobs['img1s'] - mean

        blobs['img0_nomean_resize'] = tf.image.resize_bilinear(blobs['img0_nomean'], size=[ADAPTED_HEIGHT, ADAPTED_WIDTH], align_corners=True)
        blobs['img1_nomean_resize'] = tf.image.resize_bilinear(blobs['img1_nomean'], size=[ADAPTED_HEIGHT, ADAPTED_WIDTH], align_corners=True)
        
        blobs['conv1a'] = tf.pad(blobs['img0_nomean_resize'], [[0,0], [3,3], [3,3], [0,0]])
        blobs['conv1a'] = tf.nn.conv2d(blobs['conv1a'], self.weights['conv1_w'], strides=[1,2,2,1], padding="VALID") + self.weights['conv1_b']
        blobs['conv1a'] = self.leaky_relu(blobs['conv1a'], 0.1)

        blobs['conv1b'] = tf.pad(blobs['img1_nomean_resize'], [[0,0], [3,3], [3,3], [0,0]])
        blobs['conv1b'] = tf.nn.conv2d(blobs['conv1b'], self.weights['conv1_w'], strides=[1,2,2,1], padding="VALID") + self.weights['conv1_b']
        blobs['conv1b'] = self.leaky_relu(blobs['conv1b'], 0.1)

        blobs['conv2a'] = tf.pad(blobs['conv1a'], [[0,0], [2,2], [2,2], [0,0]])
        blobs['conv2a'] = tf.nn.conv2d(blobs['conv2a'], self.weights['conv2_w'], strides=[1,2,2,1], padding="VALID") + self.weights['conv2_b']
        blobs['conv2a'] = self.leaky_relu(blobs['conv2a'], 0.1)

        blobs['conv2b'] = tf.pad(blobs['conv1b'], [[0,0], [2,2], [2,2], [0,0]])
        blobs['conv2b'] = tf.nn.conv2d(blobs['conv2b'], self.weights['conv2_w'], strides=[1,2,2,1], padding="VALID") + self.weights['conv2_b']
        blobs['conv2b'] = self.leaky_relu(blobs['conv2b'], 0.1)

        blobs['conv3a'] = tf.pad(blobs['conv2a'], [[0,0], [2,2], [2,2], [0,0]])
        blobs['conv3a'] = tf.nn.conv2d(blobs['conv3a'], self.weights['conv3_w'], strides=[1,2,2,1], padding="VALID") + self.weights['conv3_b']
        blobs['conv3a'] = self.leaky_relu(blobs['conv3a'], 0.1)

        blobs['conv3b'] = tf.pad(blobs['conv2b'], [[0,0], [2,2], [2,2], [0,0]])
        blobs['conv3b'] = tf.nn.conv2d(blobs['conv3b'], self.weights['conv3_w'], strides=[1,2,2,1], padding="VALID") + self.weights['conv3_b']
        blobs['conv3b'] = self.leaky_relu(blobs['conv3b'], 0.1)

        # this might be considered a bit hacky
        tmp = []
        x1_l = []
        x2_l = []
        for di in range(-20, 21, 2):
            for dj in range(-20, 21, 2):
                x1 = tf.pad(blobs['conv3a'], [[0,0], [20,20], [20,20], [0,0]])
                x2 = tf.pad(blobs['conv3b'], [[0,0], [20-di,20+di], [20-dj,20+dj], [0,0]])
                x1_l.append(x1)
                x2_l.append(x2)
                c = tf.nn.conv2d(x1*x2, tf.ones([1, 1, 256, 1])/256., strides=[1,1,1,1], padding='VALID')
                tmp.append(c[:,20:-20,20:-20,:])
        for i in range(len(tmp)-1):
            #self.run_after(tmp[i], tmp[i+1])
            self.run_after(x1_l[i], tmp[i+1])
            self.run_after(x2_l[i], tmp[i+1])
        blobs['corr'] = tf.concat(tmp, axis=3) 
        blobs['corr'] = self.leaky_relu(blobs['corr'], 0.1)

        blobs['conv_redir'] = tf.nn.conv2d(blobs['conv3a'], self.weights['conv_redir_w'], strides=[1,1,1,1], padding="VALID") + self.weights['conv_redir_b']
        blobs['conv_redir'] = self.leaky_relu(blobs['conv_redir'], 0.1)

        blobs['blob16'] = tf.concat([blobs['conv_redir'], blobs['corr']], axis=3)

        blobs['conv3_1'] = tf.nn.conv2d(blobs['blob16'], self.weights['conv3_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['conv3_1_b']
        blobs['conv3_1'] = self.leaky_relu(blobs['conv3_1'], 0.1)

        blobs['conv4'] = tf.pad(blobs['conv3_1'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['conv4'] = tf.nn.conv2d(blobs['conv4'], self.weights['conv4_w'], strides=[1,2,2,1], padding="VALID") + self.weights['conv4_b']
        blobs['conv4'] = self.leaky_relu(blobs['conv4'], 0.1)

        blobs['conv4_1'] = tf.nn.conv2d(blobs['conv4'], self.weights['conv4_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['conv4_1_b']
        blobs['conv4_1'] = self.leaky_relu(blobs['conv4_1'], 0.1)

        blobs['conv5'] = tf.pad(blobs['conv4_1'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['conv5'] = tf.nn.conv2d(blobs['conv5'], self.weights['conv5_w'], strides=[1,2,2,1], padding="VALID") + self.weights['conv5_b']
        blobs['conv5'] = self.leaky_relu(blobs['conv5'], 0.1)

        blobs['conv5_1'] = tf.nn.conv2d(blobs['conv5'], self.weights['conv5_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['conv5_1_b']
        blobs['conv5_1'] = self.leaky_relu(blobs['conv5_1'], 0.1)

        blobs['conv6'] = tf.pad(blobs['conv5_1'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['conv6'] = tf.nn.conv2d(blobs['conv6'], self.weights['conv6_w'], strides=[1,2,2,1], padding="VALID") + self.weights['conv6_b']
        blobs['conv6'] = self.leaky_relu(blobs['conv6'], 0.1)

        blobs['conv6_1'] = tf.nn.conv2d(blobs['conv6'], self.weights['conv6_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['conv6_1_b']
        blobs['conv6_1'] = self.leaky_relu(blobs['conv6_1'], 0.1)

        blobs['predict_flow6'] = tf.nn.conv2d(blobs['conv6_1'], self.weights['Convolution1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['Convolution1_b']

        blobs['deconv5'] = tf.nn.conv2d_transpose(blobs['conv6_1'], self.weights['deconv5_w'], output_shape=[batch_size, ADAPTED_HEIGHT/32, ADAPTED_WIDTH/32, 512], strides=[1,2,2,1]) + self.weights['deconv5_b']
        blobs['deconv5'] = self.leaky_relu(blobs['deconv5'], 0.1)
        
        blobs['upsampled_flow6_to_5'] = tf.nn.conv2d_transpose(blobs['predict_flow6'], self.weights['upsample_flow6to5_w'], output_shape=[batch_size, ADAPTED_HEIGHT/32, ADAPTED_WIDTH/32, 2], strides=[1,2,2,1]) + self.weights['upsample_flow6to5_b']
        
        blobs['concat5'] = tf.concat([blobs['conv5_1'], blobs['deconv5'], blobs['upsampled_flow6_to_5']], axis=3)

        blobs['predict_flow5'] = tf.pad(blobs['concat5'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['predict_flow5'] = tf.nn.conv2d(blobs['predict_flow5'], self.weights['Convolution2_w'], strides=[1,1,1,1], padding="VALID") + self.weights['Convolution2_b']
        
        blobs['deconv4'] = tf.nn.conv2d_transpose(blobs['concat5'], self.weights['deconv4_w'], output_shape=[batch_size, ADAPTED_HEIGHT/16, ADAPTED_WIDTH/16, 256], strides=[1,2,2,1]) + self.weights['deconv4_b']
        blobs['deconv4'] = self.leaky_relu(blobs['deconv4'], 0.1)
        
        blobs['upsampled_flow5_to_4'] = tf.nn.conv2d_transpose(blobs['predict_flow5'], self.weights['upsample_flow5to4_w'], output_shape=[batch_size, ADAPTED_HEIGHT/16, ADAPTED_WIDTH/16, 2], strides=[1,2,2,1]) + self.weights['upsample_flow5to4_b']
        
        blobs['concat4'] = tf.concat([blobs['conv4_1'], blobs['deconv4'], blobs['upsampled_flow5_to_4']], axis=3)

        blobs['predict_flow4'] = tf.nn.conv2d(blobs['concat4'], self.weights['Convolution3_w'], strides=[1,1,1,1], padding="SAME") + self.weights['Convolution3_b']
        
        blobs['deconv3'] = tf.nn.conv2d_transpose(blobs['concat4'], self.weights['deconv3_w'], output_shape=[batch_size, ADAPTED_HEIGHT/8, ADAPTED_WIDTH/8, 128], strides=[1,2,2,1]) + self.weights['deconv3_b']
        blobs['deconv3'] = self.leaky_relu(blobs['deconv3'], 0.1)
        
        blobs['upsampled_flow4_to_3'] = tf.nn.conv2d_transpose(blobs['predict_flow4'], self.weights['upsample_flow4to3_w'], output_shape=[batch_size, ADAPTED_HEIGHT/8, ADAPTED_WIDTH/8, 2], strides=[1,2,2,1]) + self.weights['upsample_flow4to3_b']
        
        blobs['concat3'] = tf.concat([blobs['conv3_1'], blobs['deconv3'], blobs['upsampled_flow4_to_3']], axis=3)

        blobs['predict_flow3'] = tf.nn.conv2d(blobs['concat3'], self.weights['Convolution4_w'], strides=[1,1,1,1], padding="SAME") + self.weights['Convolution4_b']
        
        blobs['deconv2'] = tf.nn.conv2d_transpose(blobs['concat3'], self.weights['deconv2_w'], output_shape=[batch_size, ADAPTED_HEIGHT/4, ADAPTED_WIDTH/4, 64], strides=[1,2,2,1]) + self.weights['deconv2_b']
        blobs['deconv2'] = self.leaky_relu(blobs['deconv2'], 0.1)

        blobs['upsampled_flow3_to_2'] = tf.nn.conv2d_transpose(blobs['predict_flow3'], self.weights['upsample_flow3to2_w'], output_shape=[batch_size, ADAPTED_HEIGHT/4, ADAPTED_WIDTH/4, 2], strides=[1,2,2,1]) + self.weights['upsample_flow3to2_b']

        blobs['concat2'] = tf.concat([blobs['conv2a'], blobs['deconv2'], blobs['upsampled_flow3_to_2']], axis=3)

        blobs['predict_flow2'] = tf.nn.conv2d(blobs['concat2'], self.weights['Convolution5_w'], strides=[1,1,1,1], padding="SAME") + self.weights['Convolution5_b']
        
        blobs['blob41'] = blobs['predict_flow2'] * 20.

        blobs['blob42'] = tf.image.resize_bilinear(blobs['blob41'], size=[ADAPTED_HEIGHT, ADAPTED_WIDTH], align_corners=True)
        
        blobs['blob43'] = self.warp(blobs['img1_nomean_resize'], blobs['blob42'])

        blobs['blob44'] = blobs['img0_nomean_resize'] - blobs['blob43']

        #blobs['blob45'] = tf.sqrt(1e-8+tf.reduce_sum(blobs['blob44']**2, axis=3, keep_dims=True))
        blobs['blob45'] = self.l2_norm(blobs['blob44'])

        blobs['blob46'] = 0.05*blobs['blob42']

        blobs['blob47'] = tf.concat([blobs['img0_nomean_resize'], blobs['img1_nomean_resize'], blobs['blob43'], blobs['blob46'], blobs['blob45']], axis=3)
        ####################################################################################
        ####################################################################################
        ####################################################################################
        ###################### END OF THE FIRST BRANCH #####################################
        ####################################################################################
        ####################################################################################
        ####################################################################################



        blobs['blob48'] = tf.pad(blobs['blob47'], [[0,0], [3,3], [3,3], [0,0]])
        blobs['blob48'] = tf.nn.conv2d(blobs['blob48'], self.weights['net2_conv1_w'], strides=[1,2,2,1], padding="VALID") + self.weights['net2_conv1_b']
        blobs['blob48'] = self.leaky_relu(blobs['blob48'], 0.1)

        blobs['blob49'] = tf.pad(blobs['blob48'], [[0,0], [2,2], [2, 2], [0,0]])
        blobs['blob49'] = tf.nn.conv2d(blobs['blob49'], self.weights['net2_conv2_w'], strides=[1,2,2,1], padding="VALID") + self.weights['net2_conv2_b']
        blobs['blob49'] = self.leaky_relu(blobs['blob49'], 0.1)

        blobs['blob50'] = tf.pad(blobs['blob49'], [[0,0], [2,2], [2,2], [0,0]])
        blobs['blob50'] = tf.nn.conv2d(blobs['blob50'], self.weights['net2_conv3_w'], strides=[1,2,2,1], padding="VALID") + self.weights['net2_conv3_b']
        blobs['blob50'] = self.leaky_relu(blobs['blob50'], 0.1)

        blobs['blob51'] = tf.nn.conv2d(blobs['blob50'], self.weights['net2_conv3_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net2_conv3_1_b']
        blobs['blob51'] = self.leaky_relu(blobs['blob51'], 0.1)

        blobs['blob52'] = tf.pad(blobs['blob51'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['blob52'] = tf.nn.conv2d(blobs['blob52'], self.weights['net2_conv4_w'], strides=[1,2,2,1], padding="VALID") + self.weights['net2_conv4_b']
        blobs['blob52'] = self.leaky_relu(blobs['blob52'], 0.1)

        blobs['blob53'] = tf.nn.conv2d(blobs['blob52'], self.weights['net2_conv4_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net2_conv4_1_b']
        blobs['blob53'] = self.leaky_relu(blobs['blob53'], 0.1)

        blobs['blob54'] = tf.pad(blobs['blob53'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['blob54'] = tf.nn.conv2d(blobs['blob54'], self.weights['net2_conv5_w'], strides=[1,2,2,1], padding="VALID") + self.weights['net2_conv5_b']
        blobs['blob54'] = self.leaky_relu(blobs['blob54'], 0.1)

        blobs['blob55'] = tf.nn.conv2d(blobs['blob54'], self.weights['net2_conv5_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net2_conv5_1_b']
        blobs['blob55'] = self.leaky_relu(blobs['blob55'], 0.1)

        blobs['blob56'] = tf.pad(blobs['blob55'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['blob56'] = tf.nn.conv2d(blobs['blob56'], self.weights['net2_conv6_w'], strides=[1,2,2,1], padding="VALID") + self.weights['net2_conv6_b']
        blobs['blob56'] = self.leaky_relu(blobs['blob56'], 0.1)

        blobs['blob57'] = tf.nn.conv2d(blobs['blob56'], self.weights['net2_conv6_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net2_conv6_1_b']
        blobs['blob57'] = self.leaky_relu(blobs['blob57'], 0.1)
        
        blobs['blob58'] = tf.nn.conv2d(blobs['blob57'], self.weights['net2_predict_conv6_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net2_predict_conv6_b']
        
        blobs['blob59'] = tf.nn.conv2d_transpose(blobs['blob57'], self.weights['net2_deconv5_w'], output_shape=[batch_size, ADAPTED_HEIGHT/32, ADAPTED_WIDTH/32, 512], strides=[1,2,2,1]) + self.weights['net2_deconv5_b']
        blobs['blob59'] = self.leaky_relu(blobs['blob59'], 0.1)
        
        blobs['blob60'] = tf.nn.conv2d_transpose(blobs['predict_flow6'], self.weights['net2_net2_upsample_flow6to5_w'], output_shape=[batch_size, ADAPTED_HEIGHT/32, ADAPTED_WIDTH/32, 2], strides=[1,2,2,1]) + self.weights['net2_net2_upsample_flow6to5_b']
        
        blobs['blob61'] = tf.concat([blobs['blob55'], blobs['blob59'], blobs['blob60']], axis=3)

        blobs['blob62'] = tf.nn.conv2d(blobs['blob61'], self.weights['net2_predict_conv5_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net2_predict_conv5_b']
        
        blobs['blob63'] = tf.nn.conv2d_transpose(blobs['blob61'], self.weights['net2_deconv4_w'], output_shape=[batch_size, ADAPTED_HEIGHT/16, ADAPTED_WIDTH/16, 256], strides=[1,2,2,1]) + self.weights['net2_deconv4_b']
        blobs['blob63'] = self.leaky_relu(blobs['blob63'], 0.1)
        
        blobs['blob64'] = tf.nn.conv2d_transpose(blobs['blob62'], self.weights['net2_net2_upsample_flow5to4_w'], output_shape=[batch_size, ADAPTED_HEIGHT/16, ADAPTED_WIDTH/16, 2], strides=[1,2,2,1]) + self.weights['net2_net2_upsample_flow5to4_b']
        
        blobs['blob65'] = tf.concat([blobs['blob53'], blobs['blob63'], blobs['blob64']], axis=3)
        
        blobs['blob66'] = tf.nn.conv2d(blobs['blob65'], self.weights['net2_predict_conv4_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net2_predict_conv4_b']
        
        blobs['blob67'] = tf.nn.conv2d_transpose(blobs['blob65'], self.weights['net2_deconv3_w'], output_shape=[batch_size, ADAPTED_HEIGHT/8, ADAPTED_WIDTH/8, 128], strides=[1,2,2,1]) + self.weights['net2_deconv3_b']
        blobs['blob67'] = self.leaky_relu(blobs['blob67'], 0.1)
        
        blobs['blob68'] = tf.nn.conv2d_transpose(blobs['blob66'], self.weights['net2_net2_upsample_flow4to3_w'], output_shape=[batch_size, ADAPTED_HEIGHT/8, ADAPTED_WIDTH/8, 2], strides=[1,2,2,1]) + self.weights['net2_net2_upsample_flow4to3_b']
        
        blobs['blob69'] = tf.concat([blobs['blob51'], blobs['blob67'], blobs['blob68']], axis=3)

        blobs['blob70'] = tf.nn.conv2d(blobs['blob69'], self.weights['net2_predict_conv3_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net2_predict_conv3_b']
        
        blobs['blob71'] = tf.nn.conv2d_transpose(blobs['blob69'], self.weights['net2_deconv2_w'], output_shape=[batch_size, ADAPTED_HEIGHT/4, ADAPTED_WIDTH/4, 64], strides=[1,2,2,1]) + self.weights['net2_deconv2_b']
        blobs['blob71'] = self.leaky_relu(blobs['blob71'], 0.1)

        blobs['blob72'] = tf.nn.conv2d_transpose(blobs['blob70'], self.weights['net2_net2_upsample_flow3to2_w'], output_shape=[batch_size, ADAPTED_HEIGHT/4, ADAPTED_WIDTH/4, 2], strides=[1,2,2,1]) + self.weights['net2_net2_upsample_flow3to2_b']

        blobs['blob73'] = tf.concat([blobs['blob49'], blobs['blob71'], blobs['blob72']], axis=3)

        blobs['blob74'] = tf.nn.conv2d(blobs['blob73'], self.weights['net2_predict_conv2_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net2_predict_conv2_b']
        
        blobs['blob75'] = blobs['blob74'] * 20.

        blobs['blob76'] = tf.image.resize_bilinear(blobs['blob75'], size=[ADAPTED_HEIGHT, ADAPTED_WIDTH], align_corners=True)
        
        blobs['blob77'] = self.warp(blobs['img1_nomean_resize'], blobs['blob76'])

        blobs['blob78'] = blobs['img0_nomean_resize'] - blobs['blob77']

        #blobs['blob79'] = tf.sqrt(1e-8+tf.reduce_sum(blobs['blob78']**2, axis=3, keep_dims=True))
        blobs['blob79'] = self.l2_norm(blobs['blob78'])

        blobs['blob80'] = 0.05*blobs['blob76']

        blobs['blob81'] = tf.concat([blobs['img0_nomean_resize'], blobs['img1_nomean_resize'], blobs['blob77'], blobs['blob80'], blobs['blob79']], axis=3)

        ####################################################################################
        ####################################################################################
        ####################################################################################
        ###################### END OF THE SECOND BRANCH ####################################
        ####################################################################################
        ####################################################################################
        ####################################################################################


        blobs['blob82'] = tf.pad(blobs['blob81'], [[0,0], [3,3], [3,3], [0,0]])
        blobs['blob82'] = tf.nn.conv2d(blobs['blob82'], self.weights['net3_conv1_w'], strides=[1,2,2,1], padding="VALID") + self.weights['net3_conv1_b']
        blobs['blob82'] = self.leaky_relu(blobs['blob82'], 0.1)

        blobs['blob83'] = tf.pad(blobs['blob82'], [[0,0], [2,2], [2, 2], [0,0]])
        blobs['blob83'] = tf.nn.conv2d(blobs['blob83'], self.weights['net3_conv2_w'], strides=[1,2,2,1], padding="VALID") + self.weights['net3_conv2_b']
        blobs['blob83'] = self.leaky_relu(blobs['blob83'], 0.1)

        blobs['blob84'] = tf.pad(blobs['blob83'], [[0,0], [2,2], [2,2], [0,0]])
        blobs['blob84'] = tf.nn.conv2d(blobs['blob84'], self.weights['net3_conv3_w'], strides=[1,2,2,1], padding="VALID") + self.weights['net3_conv3_b']
        blobs['blob84'] = self.leaky_relu(blobs['blob84'], 0.1)

        blobs['blob85'] = tf.nn.conv2d(blobs['blob84'], self.weights['net3_conv3_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net3_conv3_1_b']
        blobs['blob85'] = self.leaky_relu(blobs['blob85'], 0.1)

        blobs['blob86'] = tf.pad(blobs['blob85'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['blob86'] = tf.nn.conv2d(blobs['blob86'], self.weights['net3_conv4_w'], strides=[1,2,2,1], padding="VALID") + self.weights['net3_conv4_b']
        blobs['blob86'] = self.leaky_relu(blobs['blob86'], 0.1)

        blobs['blob87'] = tf.nn.conv2d(blobs['blob86'], self.weights['net3_conv4_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net3_conv4_1_b']
        blobs['blob87'] = self.leaky_relu(blobs['blob87'], 0.1)

        blobs['blob88'] = tf.pad(blobs['blob87'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['blob88'] = tf.nn.conv2d(blobs['blob88'], self.weights['net3_conv5_w'], strides=[1,2,2,1], padding="VALID") + self.weights['net3_conv5_b']
        blobs['blob88'] = self.leaky_relu(blobs['blob88'], 0.1)

        blobs['blob89'] = tf.nn.conv2d(blobs['blob88'], self.weights['net3_conv5_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net3_conv5_1_b']
        blobs['blob89'] = self.leaky_relu(blobs['blob89'], 0.1)

        blobs['blob90'] = tf.pad(blobs['blob89'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['blob90'] = tf.nn.conv2d(blobs['blob90'], self.weights['net3_conv6_w'], strides=[1,2,2,1], padding="VALID") + self.weights['net3_conv6_b']
        blobs['blob90'] = self.leaky_relu(blobs['blob90'], 0.1)

        blobs['blob91'] = tf.nn.conv2d(blobs['blob90'], self.weights['net3_conv6_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net3_conv6_1_b']
        blobs['blob91'] = self.leaky_relu(blobs['blob91'], 0.1)
        
        blobs['blob92'] = tf.nn.conv2d(blobs['blob91'], self.weights['net3_predict_conv6_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net3_predict_conv6_b']
        
        blobs['blob93'] = tf.nn.conv2d_transpose(blobs['blob91'], self.weights['net3_deconv5_w'], output_shape=[batch_size, ADAPTED_HEIGHT/32, ADAPTED_WIDTH/32, 512], strides=[1,2,2,1]) + self.weights['net3_deconv5_b']
        blobs['blob93'] = self.leaky_relu(blobs['blob93'], 0.1)
        
        blobs['blob94'] = tf.nn.conv2d_transpose(blobs['blob92'], self.weights['net3_net3_upsample_flow6to5_w'], output_shape=[batch_size, ADAPTED_HEIGHT/32, ADAPTED_WIDTH/32, 2], strides=[1,2,2,1]) + self.weights['net3_net3_upsample_flow6to5_b']
        
        blobs['blob95'] = tf.concat([blobs['blob89'], blobs['blob93'], blobs['blob94']], axis=3)

        blobs['blob96'] = tf.nn.conv2d(blobs['blob95'], self.weights['net3_predict_conv5_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net3_predict_conv5_b']
        
        blobs['blob97'] = tf.nn.conv2d_transpose(blobs['blob95'], self.weights['net3_deconv4_w'], output_shape=[batch_size, ADAPTED_HEIGHT/16, ADAPTED_WIDTH/16, 256], strides=[1,2,2,1]) + self.weights['net3_deconv4_b']
        blobs['blob97'] = self.leaky_relu(blobs['blob97'], 0.1)
        
        blobs['blob98'] = tf.nn.conv2d_transpose(blobs['blob96'], self.weights['net3_net3_upsample_flow5to4_w'], output_shape=[batch_size, ADAPTED_HEIGHT/16, ADAPTED_WIDTH/16, 2], strides=[1,2,2,1]) + self.weights['net3_net3_upsample_flow5to4_b']
        
        blobs['blob99'] = tf.concat([blobs['blob87'], blobs['blob97'], blobs['blob98']], axis=3)
        
        blobs['blob100'] = tf.nn.conv2d(blobs['blob99'], self.weights['net3_predict_conv4_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net3_predict_conv4_b']
        
        blobs['blob101'] = tf.nn.conv2d_transpose(blobs['blob99'], self.weights['net3_deconv3_w'], output_shape=[batch_size, ADAPTED_HEIGHT/8, ADAPTED_WIDTH/8, 128], strides=[1,2,2,1]) + self.weights['net3_deconv3_b']
        blobs['blob101'] = self.leaky_relu(blobs['blob101'], 0.1)
        
        blobs['blob102'] = tf.nn.conv2d_transpose(blobs['blob100'], self.weights['net3_net3_upsample_flow4to3_w'], output_shape=[batch_size, ADAPTED_HEIGHT/8, ADAPTED_WIDTH/8, 2], strides=[1,2,2,1]) + self.weights['net3_net3_upsample_flow4to3_b']
        
        blobs['blob103'] = tf.concat([blobs['blob85'], blobs['blob101'], blobs['blob102']], axis=3)

        blobs['blob104'] = tf.nn.conv2d(blobs['blob103'], self.weights['net3_predict_conv3_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net3_predict_conv3_b']
        
        blobs['blob105'] = tf.nn.conv2d_transpose(blobs['blob103'], self.weights['net3_deconv2_w'], output_shape=[batch_size, ADAPTED_HEIGHT/4, ADAPTED_WIDTH/4, 64], strides=[1,2,2,1]) + self.weights['net3_deconv2_b']
        blobs['blob105'] = self.leaky_relu(blobs['blob105'], 0.1)

        blobs['blob106'] = tf.nn.conv2d_transpose(blobs['blob104'], self.weights['net3_net3_upsample_flow3to2_w'], output_shape=[batch_size, ADAPTED_HEIGHT/4, ADAPTED_WIDTH/4, 2], strides=[1,2,2,1]) + self.weights['net3_net3_upsample_flow3to2_b']

        blobs['blob107'] = tf.concat([blobs['blob83'], blobs['blob105'], blobs['blob106']], axis=3)

        blobs['blob108'] = tf.nn.conv2d(blobs['blob107'], self.weights['net3_predict_conv2_w'], strides=[1,1,1,1], padding="SAME") + self.weights['net3_predict_conv2_b']
        
        blobs['blob109'] = blobs['blob108'] * 20.

        ####################################################################################
        ####################################################################################
        ####################################################################################
        ###################### END OF THE THIRD BRANCH  ####################################
        ####################################################################################
        ####################################################################################
        ####################################################################################

        blobs['blob110'] = tf.concat([blobs['img0_nomean_resize'], blobs['img1_nomean_resize']], axis=3)
        #self.run_after(blobs['blob110'], blobs['blob109'])
        
        blobs['blob111'] = tf.nn.conv2d(blobs['blob110'], self.weights['netsd_conv0_w'], strides=[1,1,1,1], padding="SAME") + self.weights['netsd_conv0_b']
        blobs['blob111'] = self.leaky_relu(blobs['blob111'], 0.1)

        blobs['blob112'] = tf.pad(blobs['blob111'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['blob112'] = tf.nn.conv2d(blobs['blob112'], self.weights['netsd_conv1_w'], strides=[1,2,2,1], padding="VALID") + self.weights['netsd_conv1_b']
        blobs['blob112'] = self.leaky_relu(blobs['blob112'], 0.1)

        blobs['blob113'] = tf.nn.conv2d(blobs['blob112'], self.weights['netsd_conv1_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['netsd_conv1_1_b']
        blobs['blob113'] = self.leaky_relu(blobs['blob113'], 0.1)
        
        blobs['blob114'] = tf.pad(blobs['blob113'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['blob114'] = tf.nn.conv2d(blobs['blob114'], self.weights['netsd_conv2_w'], strides=[1,2,2,1], padding="VALID") + self.weights['netsd_conv2_b']
        blobs['blob114'] = self.leaky_relu(blobs['blob114'], 0.1)

        blobs['blob115'] = tf.nn.conv2d(blobs['blob114'], self.weights['netsd_conv2_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['netsd_conv2_1_b']
        blobs['blob115'] = self.leaky_relu(blobs['blob115'], 0.1)
        
        blobs['blob116'] = tf.pad(blobs['blob115'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['blob116'] = tf.nn.conv2d(blobs['blob116'], self.weights['netsd_conv3_w'], strides=[1,2,2,1], padding="VALID") + self.weights['netsd_conv3_b']
        blobs['blob116'] = self.leaky_relu(blobs['blob116'], 0.1)

        blobs['blob117'] = tf.nn.conv2d(blobs['blob116'], self.weights['netsd_conv3_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['netsd_conv3_1_b']
        blobs['blob117'] = self.leaky_relu(blobs['blob117'], 0.1)
        
        blobs['blob118'] = tf.pad(blobs['blob117'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['blob118'] = tf.nn.conv2d(blobs['blob118'], self.weights['netsd_conv4_w'], strides=[1,2,2,1], padding="VALID") + self.weights['netsd_conv4_b']
        blobs['blob118'] = self.leaky_relu(blobs['blob118'], 0.1)

        blobs['blob119'] = tf.nn.conv2d(blobs['blob118'], self.weights['netsd_conv4_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['netsd_conv4_1_b']
        blobs['blob119'] = self.leaky_relu(blobs['blob119'], 0.1)
        
        blobs['blob120'] = tf.pad(blobs['blob119'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['blob120'] = tf.nn.conv2d(blobs['blob120'], self.weights['netsd_conv5_w'], strides=[1,2,2,1], padding="VALID") + self.weights['netsd_conv5_b']
        blobs['blob120'] = self.leaky_relu(blobs['blob120'], 0.1)

        blobs['blob121'] = tf.nn.conv2d(blobs['blob120'], self.weights['netsd_conv5_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['netsd_conv5_1_b']
        blobs['blob121'] = self.leaky_relu(blobs['blob121'], 0.1)
        
        blobs['blob122'] = tf.pad(blobs['blob121'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['blob122'] = tf.nn.conv2d(blobs['blob122'], self.weights['netsd_conv6_w'], strides=[1,2,2,1], padding="VALID") + self.weights['netsd_conv6_b']
        blobs['blob122'] = self.leaky_relu(blobs['blob122'], 0.1)

        blobs['blob123'] = tf.nn.conv2d(blobs['blob122'], self.weights['netsd_conv6_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['netsd_conv6_1_b']
        blobs['blob123'] = self.leaky_relu(blobs['blob123'], 0.1)

        blobs['blob124'] = tf.nn.conv2d(blobs['blob123'], self.weights['netsd_Convolution1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['netsd_Convolution1_b']
        
        blobs['blob125'] = tf.nn.conv2d_transpose(blobs['blob123'], self.weights['netsd_deconv5_w'], output_shape=[batch_size, ADAPTED_HEIGHT/32, ADAPTED_WIDTH/32, 512], strides=[1,2,2,1]) + self.weights['netsd_deconv5_b']
        blobs['blob125'] = self.leaky_relu(blobs['blob125'], 0.1)
        
        blobs['blob126'] = tf.nn.conv2d_transpose(blobs['blob124'], self.weights['netsd_upsample_flow6to5_w'], output_shape=[batch_size, ADAPTED_HEIGHT/32, ADAPTED_WIDTH/32, 2], strides=[1,2,2,1]) + self.weights['netsd_upsample_flow6to5_b']
        
        blobs['blob127'] = tf.concat([blobs['blob121'], blobs['blob125'], blobs['blob126']], axis=3)

        blobs['blob128'] = tf.nn.conv2d(blobs['blob127'], self.weights['netsd_interconv5_w'], strides=[1,1,1,1], padding="SAME") + self.weights['netsd_interconv5_b']
        
        blobs['blob129'] = tf.nn.conv2d(blobs['blob128'], self.weights['netsd_Convolution2_w'], strides=[1,1,1,1], padding="SAME") + self.weights['netsd_Convolution2_b']

        blobs['blob130'] = tf.nn.conv2d_transpose(blobs['blob127'], self.weights['netsd_deconv4_w'], output_shape=[batch_size, ADAPTED_HEIGHT/16, ADAPTED_WIDTH/16, 256], strides=[1,2,2,1]) + self.weights['netsd_deconv4_b']
        blobs['blob130'] = self.leaky_relu(blobs['blob130'], 0.1)
        
        blobs['blob131'] = tf.nn.conv2d_transpose(blobs['blob129'], self.weights['netsd_upsample_flow5to4_w'], output_shape=[batch_size, ADAPTED_HEIGHT/16, ADAPTED_WIDTH/16, 2], strides=[1,2,2,1]) + self.weights['netsd_upsample_flow5to4_b']
        
        blobs['blob132'] = tf.concat([blobs['blob119'], blobs['blob130'], blobs['blob131']], axis=3)
        
        blobs['blob133'] = tf.nn.conv2d(blobs['blob132'], self.weights['netsd_interconv4_w'], strides=[1,1,1,1], padding="SAME") + self.weights['netsd_interconv4_b']
        
        blobs['blob134'] = tf.nn.conv2d(blobs['blob133'], self.weights['netsd_Convolution3_w'], strides=[1,1,1,1], padding="SAME") + self.weights['netsd_Convolution3_b']

        blobs['blob135'] = tf.nn.conv2d_transpose(blobs['blob132'], self.weights['netsd_deconv3_w'], output_shape=[batch_size, ADAPTED_HEIGHT/8, ADAPTED_WIDTH/8, 128], strides=[1,2,2,1]) + self.weights['netsd_deconv3_b']
        blobs['blob135'] = self.leaky_relu(blobs['blob135'], 0.1)
        
        blobs['blob136'] = tf.nn.conv2d_transpose(blobs['blob134'], self.weights['netsd_upsample_flow4to3_w'], output_shape=[batch_size, ADAPTED_HEIGHT/8, ADAPTED_WIDTH/8, 2], strides=[1,2,2,1]) + self.weights['netsd_upsample_flow4to3_b']
        
        blobs['blob137'] = tf.concat([blobs['blob117'], blobs['blob135'], blobs['blob136']], axis=3)

        blobs['blob138'] = tf.nn.conv2d(blobs['blob137'], self.weights['netsd_interconv3_w'], strides=[1,1,1,1], padding="SAME") + self.weights['netsd_interconv3_b']
        
        blobs['blob139'] = tf.nn.conv2d(blobs['blob138'], self.weights['netsd_Convolution4_w'], strides=[1,1,1,1], padding="SAME") + self.weights['netsd_Convolution4_b']

        blobs['blob140'] = tf.nn.conv2d_transpose(blobs['blob137'], self.weights['netsd_deconv2_w'], output_shape=[batch_size, ADAPTED_HEIGHT/4, ADAPTED_WIDTH/4, 64], strides=[1,2,2,1]) + self.weights['netsd_deconv2_b']
        blobs['blob140'] = self.leaky_relu(blobs['blob140'], 0.1)

        blobs['blob141'] = tf.nn.conv2d_transpose(blobs['blob139'], self.weights['netsd_upsample_flow3to2_w'], output_shape=[batch_size, ADAPTED_HEIGHT/4, ADAPTED_WIDTH/4, 2], strides=[1,2,2,1]) + self.weights['netsd_upsample_flow3to2_b']

        blobs['blob142'] = tf.concat([blobs['blob115'], blobs['blob140'], blobs['blob141']], axis=3)

        blobs['blob143'] = tf.nn.conv2d(blobs['blob142'], self.weights['netsd_interconv2_w'], strides=[1,1,1,1], padding="SAME") + self.weights['netsd_interconv2_b']
        
        blobs['blob144'] = tf.nn.conv2d(blobs['blob143'], self.weights['netsd_Convolution5_w'], strides=[1,1,1,1], padding="SAME") + self.weights['netsd_Convolution5_b']

        blobs['blob145'] = 0.05*blobs['blob144']

        blobs['blob146'] = tf.image.resize_nearest_neighbor(blobs['blob145'], size=[ADAPTED_HEIGHT, ADAPTED_WIDTH], align_corners=False)
        
        blobs['blob147'] = tf.image.resize_nearest_neighbor(blobs['blob109'], size=[ADAPTED_HEIGHT, ADAPTED_WIDTH], align_corners=False)

        #blobs['blob148'] = tf.sqrt(1e-8+tf.reduce_sum(blobs['blob146']**2, axis=3, keep_dims=True))
        blobs['blob148'] = self.l2_norm(blobs['blob146'])

        #blobs['blob149'] = tf.sqrt(1e-8+tf.reduce_sum(blobs['blob147']**2, axis=3, keep_dims=True))
        blobs['blob149'] = self.l2_norm(blobs['blob147'])

        blobs['blob150'] = self.warp(blobs['img1_nomean_resize'], blobs['blob146'])

        blobs['blob151'] = blobs['img0_nomean_resize'] - blobs['blob150']

        #blobs['blob152'] = tf.sqrt(1e-8+tf.reduce_sum(blobs['blob151']**2, axis=3, keep_dims=True))
        blobs['blob152'] = self.l2_norm(blobs['blob151'])

        blobs['blob153'] = self.warp(blobs['img1_nomean_resize'], blobs['blob147'])

        blobs['blob154'] = blobs['img0_nomean_resize'] - blobs['blob153']

        #blobs['blob155'] = tf.sqrt(1e-8+tf.reduce_sum(blobs['blob154']**2, axis=3, keep_dims=True))
        blobs['blob155'] = self.l2_norm(blobs['blob154'])

        blobs['blob156'] = tf.concat([blobs['img0_nomean_resize'], blobs['blob146'], blobs['blob147'], blobs['blob148'], blobs['blob149'], blobs['blob152'], blobs['blob155']], axis=3)

        blobs['blob157'] = tf.nn.conv2d(blobs['blob156'], self.weights['fuse_conv0_w'], strides=[1,1,1,1], padding="SAME") + self.weights['fuse_conv0_b']
        blobs['blob157'] = self.leaky_relu(blobs['blob157'], 0.1)

        blobs['blob158'] = tf.pad(blobs['blob157'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['blob158'] = tf.nn.conv2d(blobs['blob158'], self.weights['fuse_conv1_w'], strides=[1,2,2,1], padding="VALID") + self.weights['fuse_conv1_b']
        blobs['blob158'] = self.leaky_relu(blobs['blob158'], 0.1)

        blobs['blob159'] = tf.nn.conv2d(blobs['blob158'], self.weights['fuse_conv1_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['fuse_conv1_1_b']
        blobs['blob159'] = self.leaky_relu(blobs['blob159'], 0.1)
        
        blobs['blob160'] = tf.pad(blobs['blob159'], [[0,0], [1,1], [1,1], [0,0]])
        blobs['blob160'] = tf.nn.conv2d(blobs['blob160'], self.weights['fuse_conv2_w'], strides=[1,2,2,1], padding="VALID") + self.weights['fuse_conv2_b']
        blobs['blob160'] = self.leaky_relu(blobs['blob160'], 0.1)

        blobs['blob161'] = tf.nn.conv2d(blobs['blob160'], self.weights['fuse_conv2_1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['fuse_conv2_1_b']
        blobs['blob161'] = self.leaky_relu(blobs['blob161'], 0.1)
        
        blobs['blob162'] = tf.nn.conv2d(blobs['blob161'], self.weights['fuse__Convolution5_w'], strides=[1,1,1,1], padding="SAME") + self.weights['fuse__Convolution5_b']
        
        blobs['blob163'] = tf.nn.conv2d_transpose(blobs['blob161'], self.weights['fuse_deconv1_w'], output_shape=[batch_size, ADAPTED_HEIGHT/2, ADAPTED_WIDTH/2, 32], strides=[1,2,2,1]) + self.weights['fuse_deconv1_b']
        blobs['blob163'] = self.leaky_relu(blobs['blob163'], 0.1)

        blobs['blob164'] = tf.nn.conv2d_transpose(blobs['blob162'], self.weights['fuse_upsample_flow2to1_w'], output_shape=[batch_size, ADAPTED_HEIGHT/2, ADAPTED_WIDTH/2, 2], strides=[1,2,2,1]) + self.weights['fuse_upsample_flow2to1_b']

        blobs['blob165'] = tf.concat([blobs['blob159'], blobs['blob163'], blobs['blob164']], axis=3)

        blobs['blob166'] = tf.nn.conv2d(blobs['blob165'], self.weights['fuse_interconv1_w'], strides=[1,1,1,1], padding="SAME") + self.weights['fuse_interconv1_b']
        
        blobs['blob167'] = tf.nn.conv2d(blobs['blob166'], self.weights['fuse__Convolution6_w'], strides=[1,1,1,1], padding="SAME") + self.weights['fuse__Convolution6_b']

        blobs['blob168'] = tf.nn.conv2d_transpose(blobs['blob165'], self.weights['fuse_deconv0_w'], output_shape=[batch_size, ADAPTED_HEIGHT/1, ADAPTED_WIDTH/1, 16], strides=[1,2,2,1]) + self.weights['fuse_deconv0_b']
        blobs['blob168'] = self.leaky_relu(blobs['blob168'], 0.1)

        blobs['blob169'] = tf.nn.conv2d_transpose(blobs['blob167'], self.weights['fuse_upsample_flow1to0_w'], output_shape=[batch_size, ADAPTED_HEIGHT, ADAPTED_WIDTH, 2], strides=[1,2,2,1]) + self.weights['fuse_upsample_flow1to0_b']

        blobs['blob170'] = tf.concat([blobs['blob157'], blobs['blob168'], blobs['blob169']], axis=3)

        blobs['blob171'] = tf.nn.conv2d(blobs['blob170'], self.weights['fuse_interconv0_w'], strides=[1,1,1,1], padding="SAME") + self.weights['fuse_interconv0_b']
        
        blobs['blob172'] = tf.nn.conv2d(blobs['blob171'], self.weights['fuse__Convolution7_w'], strides=[1,1,1,1], padding="SAME") + self.weights['fuse__Convolution7_b']

        blobs['predict_flow_resize'] = tf.image.resize_bilinear(blobs['blob172'], size=[TARGET_HEIGHT, TARGET_WIDTH], align_corners=True)

        scale = tf.stack([SCALE_WIDTH, SCALE_HEIGHT])
        scale = tf.reshape(scale, [1,1,1,2])
        blobs['predict_flow_final'] = scale*blobs['predict_flow_resize']
        
        self.blobs = blobs

        return blobs

    # very beautiful code
    def all_variables(self):
        return [('netsd_deconv5_w', (4, 4, 512, 1024)),
            ('netsd_conv1_b', (64,)),
            ('netsd_upsample_flow5to4_w', (4, 4, 2, 2)),
            ('conv2_b', (128,)),
            ('fuse__Convolution5_w', (3, 3, 128, 2)),
            ('netsd_conv4_1_w', (3, 3, 512, 512)),
            ('netsd_interconv3_w', (3, 3, 386, 128)),
            ('netsd_deconv4_w', (4, 4, 256, 1026)),
            ('deconv4_b', (256,)),
            ('fuse_interconv0_w', (3, 3, 82, 16)),
            ('netsd_Convolution2_b', (2,)),
            ('net3_conv4_b', (512,)),
            ('net3_conv3_b', (256,)),
            ('net3_predict_conv2_w', (3, 3, 194, 2)),
            ('net3_predict_conv3_b', (2,)),
            ('conv6_1_w', (3, 3, 1024, 1024)),
            ('fuse_upsample_flow2to1_b', (2,)),
            ('Convolution1_w', (3, 3, 1024, 2)),
            ('net3_deconv3_w', (4, 4, 128, 770)),
            ('net2_deconv3_b', (128,)),
            ('fuse_conv1_w', (3, 3, 64, 64)),
            ('conv5_w', (3, 3, 512, 512)),
            ('Convolution4_w', (3, 3, 386, 2)),
            ('fuse_conv0_b', (64,)),
            ('net2_conv3_w', (5, 5, 128, 256)),
            ('upsample_flow4to3_b', (2,)),
            ('netsd_conv4_1_b', (512,)),
            ('fuse_upsample_flow2to1_w', (4, 4, 2, 2)),
            ('netsd_conv4_b', (512,)),
            ('net2_net2_upsample_flow3to2_b', (2,)),
            ('net3_predict_conv4_b', (2,)),
            ('fuse_upsample_flow1to0_b', (2,)),
            ('conv4_1_w', (3, 3, 512, 512)),
            ('deconv2_b', (64,)),
            ('net2_conv4_1_w', (3, 3, 512, 512)),
            ('net3_deconv4_w', (4, 4, 256, 1026)),
            ('net2_deconv5_b', (512,)),
            ('netsd_deconv5_b', (512,)),
            ('net2_deconv2_b', (64,)),
            ('net3_conv2_b', (128,)),
            ('conv_redir_w', (1, 1, 256, 32)),
            ('fuse_conv1_1_b', (128,)),
            ('net2_deconv5_w', (4, 4, 512, 1024)),
            ('net2_conv5_b', (512,)),
            ('net2_conv4_w', (3, 3, 256, 512)),
            ('net2_predict_conv6_w', (3, 3, 1024, 2)),
            ('netsd_conv5_b', (512,)),
            ('deconv4_w', (4, 4, 256, 1026)),
            ('net2_net2_upsample_flow4to3_b', (2,)),
            ('fuse__Convolution6_w', (3, 3, 32, 2)),
            ('net3_deconv2_w', (4, 4, 64, 386)),
            ('net2_conv6_1_w', (3, 3, 1024, 1024)),
            ('netsd_conv0_b', (64,)),
            ('netsd_conv5_1_w', (3, 3, 512, 512)),
            ('net2_conv6_1_b', (1024,)),
            ('net3_conv2_w', (5, 5, 64, 128)),
            ('net3_predict_conv6_w', (3, 3, 1024, 2)),
            ('net3_conv4_1_b', (512,)),
            ('net3_net3_upsample_flow4to3_w', (4, 4, 2, 2)),
            ('net2_deconv2_w', (4, 4, 64, 386)),
            ('deconv3_b', (128,)),
            ('netsd_interconv5_b', (512,)),
            ('net2_conv3_1_w', (3, 3, 256, 256)),
            ('netsd_interconv4_w', (3, 3, 770, 256)),
            ('net3_deconv3_b', (128,)),
            ('fuse_conv0_w', (3, 3, 11, 64)),
            ('net3_predict_conv6_b', (2,)),
            ('fuse_upsample_flow1to0_w', (4, 4, 2, 2)),
            ('netsd_deconv3_b', (128,)),
            ('net3_predict_conv5_w', (3, 3, 1026, 2)),
            ('netsd_conv5_w', (3, 3, 512, 512)),
            ('netsd_interconv5_w', (3, 3, 1026, 512)),
            ('netsd_Convolution3_w', (3, 3, 256, 2)),
            ('net2_predict_conv4_w', (3, 3, 770, 2)),
            ('deconv2_w', (4, 4, 64, 386)),
            ('net3_predict_conv5_b', (2,)),
            ('fuse__Convolution5_b', (2,)),
            ('fuse__Convolution7_w', (3, 3, 16, 2)),
            ('net2_net2_upsample_flow6to5_w', (4, 4, 2, 2)),
            ('netsd_conv3_b', (256,)),
            ('net3_conv6_w', (3, 3, 512, 1024)),
            ('net3_conv1_b', (64,)),
            ('netsd_Convolution4_b', (2,)),
            ('net3_conv3_w', (5, 5, 128, 256)),
            ('netsd_conv0_w', (3, 3, 6, 64)),
            ('net2_conv4_b', (512,)),
            ('net2_predict_conv3_w', (3, 3, 386, 2)),
            ('net3_net3_upsample_flow3to2_w', (4, 4, 2, 2)),
            ('fuse_conv1_1_w', (3, 3, 64, 128)),
            ('deconv5_b', (512,)),
            ('fuse__Convolution7_b', (2,)),
            ('net3_conv6_1_w', (3, 3, 1024, 1024)),
            ('net3_net3_upsample_flow5to4_w', (4, 4, 2, 2)),
            ('net3_conv4_w', (3, 3, 256, 512)),
            ('upsample_flow5to4_w', (4, 4, 2, 2)),
            ('conv4_1_b', (512,)),
            ('img0s_aug_b', (320, 448, 3, 1)),
            ('conv5_1_b', (512,)),
            ('net3_conv4_1_w', (3, 3, 512, 512)),
            ('upsample_flow5to4_b', (2,)),
            ('net3_conv3_1_b', (256,)),
            ('Convolution1_b', (2,)),
            ('upsample_flow4to3_w', (4, 4, 2, 2)),
            ('conv5_1_w', (3, 3, 512, 512)),
            ('conv3_1_b', (256,)),
            ('conv3_w', (5, 5, 128, 256)),
            ('net2_conv2_b', (128,)),
            ('net3_net3_upsample_flow6to5_w', (4, 4, 2, 2)),
            ('upsample_flow3to2_b', (2,)),
            ('netsd_Convolution5_w', (3, 3, 64, 2)),
            ('netsd_interconv2_w', (3, 3, 194, 64)),
            ('net2_predict_conv6_b', (2,)),
            ('net2_deconv4_w', (4, 4, 256, 1026)),
            ('scale_conv1_b', (2,)),
            ('net2_net2_upsample_flow5to4_w', (4, 4, 2, 2)),
            ('netsd_conv2_b', (128,)),
            ('netsd_conv2_1_b', (128,)),
            ('netsd_upsample_flow6to5_w', (4, 4, 2, 2)),
            ('net2_predict_conv5_b', (2,)),
            ('net3_conv6_1_b', (1024,)),
            ('netsd_conv6_w', (3, 3, 512, 1024)),
            ('Convolution4_b', (2,)),
            ('net2_predict_conv4_b', (2,)),
            ('fuse_deconv1_b', (32,)),
            ('conv3_1_w', (3, 3, 473, 256)),
            ('net3_deconv2_b', (64,)),
            ('netsd_conv6_b', (1024,)),
            ('net2_conv5_1_w', (3, 3, 512, 512)),
            ('net3_conv5_1_w', (3, 3, 512, 512)),
            ('deconv5_w', (4, 4, 512, 1024)),
            ('fuse_conv2_b', (128,)),
            ('netsd_conv1_1_b', (128,)),
            ('netsd_upsample_flow6to5_b', (2,)),
            ('Convolution5_w', (3, 3, 194, 2)),
            ('scale_conv1_w', (1, 1, 2, 2)),
            ('net2_net2_upsample_flow5to4_b', (2,)),
            ('conv6_1_b', (1024,)),
            ('fuse_conv2_1_b', (128,)),
            ('netsd_Convolution5_b', (2,)),
            ('netsd_conv3_1_b', (256,)),
            ('conv2_w', (5, 5, 64, 128)),
            ('fuse_conv2_w', (3, 3, 128, 128)),
            ('net2_conv2_w', (5, 5, 64, 128)),
            ('conv3_b', (256,)),
            ('net3_deconv5_w', (4, 4, 512, 1024)),
            ('img1s_aug_w', (1, 1, 1, 1)),
            ('netsd_conv2_w', (3, 3, 128, 128)),
            ('conv6_w', (3, 3, 512, 1024)),
            ('netsd_conv4_w', (3, 3, 256, 512)),
            ('net2_conv1_w', (7, 7, 12, 64)),
            ('netsd_Convolution1_w', (3, 3, 1024, 2)),
            ('netsd_conv1_w', (3, 3, 64, 64)),
            ('netsd_deconv4_b', (256,)),
            ('conv4_w', (3, 3, 256, 512)),
            ('conv5_b', (512,)),
            ('net3_deconv5_b', (512,)),
            ('netsd_interconv3_b', (128,)),
            ('net3_conv3_1_w', (3, 3, 256, 256)),
            ('net2_predict_conv5_w', (3, 3, 1026, 2)),
            ('Convolution3_b', (2,)),
            ('netsd_conv5_1_b', (512,)),
            ('netsd_interconv4_b', (256,)),
            ('conv4_b', (512,)),
            ('net3_net3_upsample_flow6to5_b', (2,)),
            ('Convolution5_b', (2,)),
            ('fuse_conv2_1_w', (3, 3, 128, 128)),
            ('net3_net3_upsample_flow4to3_b', (2,)),
            ('conv1_w', (7, 7, 3, 64)),
            ('upsample_flow6to5_b', (2,)),
            ('conv6_b', (1024,)),
            ('netsd_upsample_flow3to2_w', (4, 4, 2, 2)),
            ('net2_deconv3_w', (4, 4, 128, 770)),
            ('netsd_conv2_1_w', (3, 3, 128, 128)),
            ('netsd_Convolution3_b', (2,)),
            ('netsd_upsample_flow4to3_w', (4, 4, 2, 2)),
            ('fuse_interconv1_w', (3, 3, 162, 32)),
            ('netsd_upsample_flow4to3_b', (2,)),
            ('netsd_conv3_1_w', (3, 3, 256, 256)),
            ('netsd_deconv3_w', (4, 4, 128, 770)),
            ('net3_conv5_b', (512,)),
            ('net3_conv5_1_b', (512,)),
            ('net2_net2_upsample_flow4to3_w', (4, 4, 2, 2)),
            ('net2_net2_upsample_flow3to2_w', (4, 4, 2, 2)),
            ('net2_conv3_b', (256,)),
            ('netsd_conv6_1_w', (3, 3, 1024, 1024)),
            ('fuse_deconv0_b', (16,)),
            ('net2_predict_conv2_w', (3, 3, 194, 2)),
            ('net2_conv1_b', (64,)),
            ('net2_conv6_b', (1024,)),
            ('net3_predict_conv2_b', (2,)),
            ('net2_conv4_1_b', (512,)),
            ('netsd_Convolution4_w', (3, 3, 128, 2)),
            ('deconv3_w', (4, 4, 128, 770)),
            ('fuse_deconv1_w', (4, 4, 32, 128)),
            ('netsd_Convolution2_w', (3, 3, 512, 2)),
            ('netsd_Convolution1_b', (2,)),
            ('net2_conv3_1_b', (256,)),
            ('fuse_conv1_b', (64,)),
            ('net2_deconv4_b', (256,)),
            ('net3_predict_conv4_w', (3, 3, 770, 2)),
            ('Convolution3_w', (3, 3, 770, 2)),
            ('netsd_upsample_flow3to2_b', (2,)),
            ('net3_net3_upsample_flow3to2_b', (2,)),
            ('fuse_interconv0_b', (16,)),
            ('Convolution2_w', (3, 3, 1026, 2)),
            ('net2_conv6_w', (3, 3, 512, 1024)),
            ('netsd_conv3_w', (3, 3, 128, 256)),
            ('netsd_upsample_flow5to4_b', (2,)),
            ('net3_predict_conv3_w', (3, 3, 386, 2)),
            ('conv_redir_b', (32,)),
            ('net2_conv5_1_b', (512,)),
            ('upsample_flow6to5_w', (4, 4, 2, 2)),
            ('net2_net2_upsample_flow6to5_b', (2,)),
            ('net3_conv6_b', (1024,)),
            ('fuse__Convolution6_b', (2,)),
            ('Convolution2_b', (2,)),
            ('upsample_flow3to2_w', (4, 4, 2, 2)),
            ('net3_conv1_w', (7, 7, 12, 64)),
            ('fuse_deconv0_w', (4, 4, 16, 162)),
            ('img0s_aug_w', (1, 1, 1, 1)),
            ('netsd_conv1_1_w', (3, 3, 64, 128)),
            ('netsd_deconv2_b', (64,)),
            ('net2_conv5_w', (3, 3, 512, 512)),
            ('fuse_interconv1_b', (32,)),
            ('netsd_conv6_1_b', (1024,)),
            ('netsd_interconv2_b', (64,)),
            ('img1s_aug_b', (320, 448, 3, 1)),
            ('netsd_deconv2_w', (4, 4, 64, 386)),
            ('net2_predict_conv3_b', (2,)),
            ('net2_predict_conv2_b', (2,)),
            ('net3_deconv4_b', (256,)),
            ('net3_net3_upsample_flow5to4_b', (2,)),
            ('conv1_b', (64,)),
            ('net3_conv5_w', (3, 3, 512, 512))]