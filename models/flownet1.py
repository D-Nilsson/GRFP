import tensorflow as tf
import glob
import numpy as np

class Flownet1:
    def __init__(self):
        self.weights = {
            'conv1': tf.Variable(tf.zeros([7, 7, 6, 64], dtype=tf.float32), name='conv1_w'),
            'conv2': tf.Variable(tf.zeros([5, 5, 64, 128], dtype=tf.float32), name='conv2_w'),
            'conv3_1': tf.Variable(tf.zeros([3, 3, 256, 256], dtype=tf.float32), name='conv3_1_w'),
            'conv3': tf.Variable(tf.zeros([5, 5, 128, 256], dtype=tf.float32), name='conv3_w'),
            'conv4': tf.Variable(tf.zeros([3, 3, 256, 512], dtype=tf.float32), name='conv4_w'),
            'conv4_1': tf.Variable(tf.zeros([3, 3, 512, 512], dtype=tf.float32), name='conv4_1_w'),
            'conv5': tf.Variable(tf.zeros([3, 3, 512, 512], dtype=tf.float32), name='conv5_w'),
            'conv5_1': tf.Variable(tf.zeros([3, 3, 512, 512], dtype=tf.float32), name='conv5_1_w'),
            'conv6': tf.Variable(tf.zeros([3, 3, 512, 1024], dtype=tf.float32), name='conv6_w'),
            'conv6_1': tf.Variable(tf.zeros([3, 3, 1024, 1024], dtype=tf.float32), name='conv6_1_w'),
            
            'Convolution1': tf.Variable(tf.zeros([3, 3, 1024, 2], dtype=tf.float32), name='Convolution1_w'),
            'Convolution2': tf.Variable(tf.zeros([3, 3, 1026, 2], dtype=tf.float32), name='Convolution2_w'),
            'Convolution3': tf.Variable(tf.zeros([3, 3, 770, 2], dtype=tf.float32), name='Convolution3_w'),
            'Convolution4': tf.Variable(tf.zeros([3, 3, 386, 2], dtype=tf.float32), name='Convolution4_w'),
            'Convolution5': tf.Variable(tf.zeros([3, 3, 194, 2], dtype=tf.float32), name='Convolution5_w'),
            'Convolution6': tf.Variable(tf.zeros([1, 1, 2, 2], dtype=tf.float32), name='Convolution6_w'),
            
            'deconv2': tf.Variable(tf.zeros([4, 4, 64, 386], dtype=tf.float32), name='deconv2_w'),
            'deconv3': tf.Variable(tf.zeros([4, 4, 128, 770], dtype=tf.float32), name='deconv3_w'),
            'deconv4': tf.Variable(tf.zeros([4, 4, 256, 1026], dtype=tf.float32), name='deconv4_w'),
            'deconv5': tf.Variable(tf.zeros([4, 4, 512, 1024], dtype=tf.float32), name='deconv5_w'),
            
            'upsample_flow3to2': tf.Variable(tf.zeros([4, 4, 2, 2], dtype=tf.float32), name='upsample_flow3to2_w'),
            'upsample_flow4to3': tf.Variable(tf.zeros([4, 4, 2, 2], dtype=tf.float32), name='upsample_flow4to3_w'),
            'upsample_flow5to4': tf.Variable(tf.zeros([4, 4, 2, 2], dtype=tf.float32), name='upsample_flow5to4_w'),
            'upsample_flow6to5': tf.Variable(tf.zeros([4, 4, 2, 2], dtype=tf.float32), name='upsample_flow6to5_w'),
        }

        self.bias = {
            'conv1': tf.Variable(tf.zeros([64], dtype=tf.float32), name='conv1_b'),
            'conv2': tf.Variable(tf.zeros([128], dtype=tf.float32), name='conv2_b'),
            'conv3_1': tf.Variable(tf.zeros([256], dtype=tf.float32), name='conv3_1_b'),
            'conv3': tf.Variable(tf.zeros([256], dtype=tf.float32), name='conv3_b'),
            'conv4': tf.Variable(tf.zeros([512], dtype=tf.float32), name='conv4_b'),
            'conv4_1': tf.Variable(tf.zeros([512], dtype=tf.float32), name='conv4_1_b'),
            'conv5': tf.Variable(tf.zeros([512], dtype=tf.float32), name='conv5_b'),
            'conv5_1': tf.Variable(tf.zeros([512], dtype=tf.float32), name='conv5_1_b'),
            'conv6': tf.Variable(tf.zeros([1024], dtype=tf.float32), name='conv6_b'),
            'conv6_1': tf.Variable(tf.zeros([1024], dtype=tf.float32), name='conv6_1_b'),
            
            'Convolution1': tf.Variable(tf.zeros([2], dtype=tf.float32), name='Convolution1_b'),
            'Convolution2': tf.Variable(tf.zeros([2], dtype=tf.float32), name='Convolution2_b'),
            'Convolution3': tf.Variable(tf.zeros([2], dtype=tf.float32), name='Convolution3_b'),
            'Convolution4': tf.Variable(tf.zeros([2], dtype=tf.float32), name='Convolution4_b'),
            'Convolution5': tf.Variable(tf.zeros([2], dtype=tf.float32), name='Convolution5_b'),
            'Convolution6': tf.Variable(tf.zeros([2], dtype=tf.float32), name='Convolution6_b'),
            
            'deconv2': tf.Variable(tf.zeros([64], dtype=tf.float32), name='deconv2_b'),
            'deconv3': tf.Variable(tf.zeros([128], dtype=tf.float32), name='deconv3_b'),
            'deconv4': tf.Variable(tf.zeros([256], dtype=tf.float32), name='deconv4_b'),
            'deconv5': tf.Variable(tf.zeros([512], dtype=tf.float32), name='deconv5_b'),
            
            'upsample_flow3to2': tf.Variable(tf.zeros([2], dtype=tf.float32), name='upsample_flow3to2_b'),
            'upsample_flow4to3': tf.Variable(tf.zeros([2], dtype=tf.float32), name='upsample_flow4to3_b'),
            'upsample_flow5to4': tf.Variable(tf.zeros([2], dtype=tf.float32), name='upsample_flow5to4_b'),
            'upsample_flow6to5': tf.Variable(tf.zeros([2], dtype=tf.float32), name='upsample_flow6to5_b'),
        }

    def get_output_tensor(self, img0, img1, out_size):
        rescaling = 0.0039216
        mean = np.array([0.411451, 0.432060, 0.450141])
        
        img0_aug = (img0 * rescaling) - mean
        img1_aug = (img1 * rescaling) - mean
        img0_nomean_resize = img0_aug
        img1_nomean_resize = img1_aug
        input_ = tf.concat(axis=3, values=[img0_nomean_resize, img1_nomean_resize])
        self.input_ = input_

        conv1 = tf.pad(input_, [[0,0], [3,3], [3,3], [0,0]])
        conv1 = tf.nn.conv2d(conv1, self.weights['conv1'], strides=[1,2,2,1], padding="VALID") + self.bias['conv1']
        conv1 = tf.maximum(conv1, 0.1*conv1) # leaky relu with negative slope 0.1
        self.conv1 = conv1

        conv2 = tf.pad(conv1, [[0,0], [2,2], [2,2], [0,0]])
        conv2 = tf.nn.conv2d(conv2, self.weights['conv2'], strides=[1,2,2,1], padding="VALID") + self.bias['conv2']
        conv2 = tf.maximum(conv2, 0.1*conv2) # leaky relu with negative slope 0.1
        self.conv2 = conv2
        
        conv3 = tf.pad(conv2, [[0,0], [2,2], [2,2], [0,0]])
        conv3 = tf.nn.conv2d(conv3, self.weights['conv3'], strides=[1,2,2,1], padding="VALID") + self.bias['conv3']
        conv3 = tf.maximum(conv3, 0.1*conv3) # leaky relu with negative slope 0.1
        self.conv3 = conv3
        
        conv3_1 = tf.nn.conv2d(conv3, self.weights['conv3_1'], strides=[1,1,1,1], padding="SAME") + self.bias['conv3_1']
        conv3_1 = tf.maximum(conv3_1, 0.1*conv3_1) # leaky relu with negative slope 0.1
        self.conv3_1 = conv3_1

        conv4 = tf.pad(conv3_1, [[0,0], [1,1], [1,1], [0,0]])
        conv4 = tf.nn.conv2d(conv4, self.weights['conv4'], strides=[1,2,2,1], padding="VALID") + self.bias['conv4']
        conv4 = tf.maximum(conv4, 0.1*conv4) # leaky relu with negative slope 0.1
        self.conv4 = conv4

        conv4_1 = tf.nn.conv2d(conv4, self.weights['conv4_1'], strides=[1,1,1,1], padding="SAME") + self.bias['conv4_1']
        conv4_1 = tf.maximum(conv4_1, 0.1*conv4_1) # leaky relu with negative slope 0.1
        self.conv4_1 = conv4_1

        conv5 = tf.pad(conv4_1, [[0,0], [1,1], [1,1], [0,0]])
        conv5 = tf.nn.conv2d(conv5, self.weights['conv5'], strides=[1,2,2,1], padding="VALID") + self.bias['conv5']
        conv5 = tf.maximum(conv5, 0.1*conv5) # leaky relu with negative slope 0.1
        self.conv5 = conv5

        conv5_1 = tf.nn.conv2d(conv5, self.weights['conv5_1'], strides=[1,1,1,1], padding="SAME") + self.bias['conv5_1']
        conv5_1 = tf.maximum(conv5_1, 0.1*conv5_1) # leaky relu with negative slope 0.1
        self.conv5_1 = conv5_1
        
        conv6 = tf.pad(conv5_1, [[0,0], [1,1], [1,1], [0,0]])
        conv6 = tf.nn.conv2d(conv6, self.weights['conv6'], strides=[1,2,2,1], padding="VALID") + self.bias['conv6']
        conv6 = tf.maximum(conv6, 0.1*conv6) # leaky relu with negative slope 0.1
        self.conv6 = conv6

        conv6_1 = tf.nn.conv2d(conv6, self.weights['conv6_1'], strides=[1,1,1,1], padding="SAME") + self.bias['conv6_1']
        conv6_1 = tf.maximum(conv6_1, 0.1*conv6_1) # leaky relu with negative slope 0.1
        self.conv6_1 = conv6_1
        
        predict_flow6 = tf.pad(conv6_1, [[0,0], [1,1], [1,1], [0,0]])
        predict_flow6 = tf.nn.conv2d(predict_flow6, self.weights['Convolution1'], strides=[1,1,1,1], padding="VALID") + self.bias['Convolution1']
        self.predict_flow6 = predict_flow6


        deconv5 = tf.nn.conv2d_transpose(conv6_1, self.weights['deconv5'], output_shape=[1, out_size[0]/32, out_size[1]/32, 512], strides=[1,2,2,1]) + self.bias['deconv5']
        deconv5 = tf.maximum(deconv5, 0.1*deconv5) # leaky relu with negative slope 0.1
        self.deconv5 = deconv5

        upsampled_flow6_to_5 = tf.nn.conv2d_transpose(predict_flow6, self.weights['upsample_flow6to5'], output_shape=[1, out_size[0]/32, out_size[1]/32, 2], strides=[1,2,2,1]) + self.bias['upsample_flow6to5']
        self.upsampled_flow6_to_5 = upsampled_flow6_to_5

        concat5 = tf.concat(axis=3, values=[conv5_1, deconv5, upsampled_flow6_to_5])
        self.concat5 = concat5

        predict_flow5 = tf.pad(concat5, [[0,0], [1,1], [1,1], [0,0]])
        predict_flow5 = tf.nn.conv2d(predict_flow5, self.weights['Convolution2'], strides=[1,1,1,1], padding="VALID") + self.bias['Convolution2']
        self.predict_flow5 = predict_flow5

        deconv4 = tf.nn.conv2d_transpose(concat5, self.weights['deconv4'], output_shape=[1, out_size[0]/16, out_size[1]/16, 256], strides=[1,2,2,1]) + self.bias['deconv4']
        deconv4 = tf.maximum(deconv4, 0.1*deconv4) # leaky relu with negative slope 0.1
        self.deconv4 = deconv4

        upsampled_flow5_to_4 = tf.nn.conv2d_transpose(predict_flow5, self.weights['upsample_flow5to4'], output_shape=[1, out_size[0]/16, out_size[1]/16, 2], strides=[1,2,2,1]) + self.bias['upsample_flow5to4']
        self.upsampled_flow5_to_4 = upsampled_flow5_to_4

        concat4 = tf.concat(axis=3, values=[conv4_1, deconv4, upsampled_flow5_to_4])
        self.concat4 = concat4

        predict_flow4 = tf.nn.conv2d(concat4, self.weights['Convolution3'], strides=[1,1,1,1], padding="SAME") + self.bias['Convolution3']
        self.predict_flow4 = predict_flow4

        deconv3 = tf.nn.conv2d_transpose(concat4, self.weights['deconv3'], output_shape=[1, out_size[0]/8, out_size[1]/8, 128], strides=[1,2,2,1]) + self.bias['deconv3']
        deconv3 = tf.maximum(deconv3, 0.1*deconv3) # leaky relu with negative slope 0.1
        self.deconv3 = deconv3

        upsampled_flow4_to_3 = tf.nn.conv2d_transpose(predict_flow4, self.weights['upsample_flow4to3'], output_shape=[1, out_size[0]/8, out_size[1]/8, 2], strides=[1,2,2,1]) + self.bias['upsample_flow4to3']
        self.upsampled_flow4_to_3 = upsampled_flow4_to_3

        concat3 = tf.concat(axis=3, values=[conv3_1, deconv3, upsampled_flow4_to_3])
        self.concat3 = concat3

        predict_flow3 = tf.nn.conv2d(concat3, self.weights['Convolution4'], strides=[1,1,1,1], padding="SAME") + self.bias['Convolution4']
        self.predict_flow3 = predict_flow3

        deconv2 = tf.nn.conv2d_transpose(concat3, self.weights['deconv2'], output_shape=[1, out_size[0]/4, out_size[1]/4, 64], strides=[1,2,2,1]) + self.bias['deconv2']
        deconv2 = tf.maximum(deconv2, 0.1*deconv2) # leaky relu with negative slope 0.1
        self.deconv2 = deconv2

        upsampled_flow3_to_2 = tf.nn.conv2d_transpose(predict_flow3, self.weights['upsample_flow3to2'], output_shape=[1, out_size[0]/4, out_size[1]/4, 2], strides=[1,2,2,1]) + self.bias['upsample_flow3to2']
        self.upsampled_flow3_to_2 = upsampled_flow3_to_2


        concat2 = tf.concat(axis=3, values=[conv2, deconv2, upsampled_flow3_to_2])
        self.concat2 = concat2

        predict_flow2 = tf.nn.conv2d(concat2, self.weights['Convolution5'], strides=[1,1,1,1], padding="SAME") + self.bias['Convolution5']
        self.predict_flow2 = predict_flow2

        blob44 = predict_flow2 * 20.0
        self.blob44 = blob44

        predict_flow_resize = tf.image.resize_bilinear(blob44, out_size, align_corners=True)

        predict_flow_final = tf.nn.conv2d(predict_flow_resize, self.weights['Convolution6'], strides=[1,1,1,1], padding="SAME") + self.bias['Convolution6']
        self.predict_flow_final = predict_flow_final
        return predict_flow_final
