import numpy as np
import tensorflow as tf

class STGRU:
    def __init__(self, tensor_size, conv_sizes, bilinear_warping_module):
        # tensor_size is something like 19 x 512 x 512
        # conv sizes are e.g. 5 x 5
        self.bilinear_warping_module = bilinear_warping_module
        channels, height, width = tensor_size
        conv_height, conv_width = conv_sizes
        conv_pad = conv_height / 2

        self.channels, self.height, self.width = channels, height, width
        self.conv_height, self.conv_width = conv_height, conv_width

        identity = np.zeros((conv_height, conv_width, channels, channels))
        for k in range(channels):
          identity[conv_height/2, conv_width/2, k, k] = 1.
        identity_map = tf.constant(identity, dtype=tf.float32)
        # identity + noise was needed for some variables to train the model 
        self.weights = {
            'ir': tf.Variable(tf.random_normal([conv_height, conv_width, 3, 1], stddev=0.001), name="W_ir"),
            'xh': tf.Variable(6.*identity_map + tf.random_normal([conv_height, conv_width, channels, channels], stddev=0.01), name="W_xh"),
            'hh': tf.Variable(6.*identity_map + tf.random_normal([conv_height, conv_width, channels, channels], stddev=0.01), name="W_hh"),
            'xz': tf.Variable(tf.random_normal([conv_height, conv_width, channels, 1], stddev=0.01), name="W_xz"),
            'hz': tf.Variable(tf.random_normal([conv_height, conv_width, channels, 1], stddev=0.01), name="W_hz"),
            'lambda': tf.Variable(tf.constant(2., dtype=tf.float32), name="lambda"),
            'bias_r': tf.Variable(tf.zeros([1], dtype=tf.float32), name="bias_r"),
            'bias_z': tf.Variable(tf.zeros([channels], dtype=tf.float32), name="bias_z"),
        }

    def get_one_step_predictor(self):
        input_images_tensor = tf.placeholder('float', [2, 1, self.height, self.width, 3], name="gru_input_images")
        input_images = tf.unstack(input_images_tensor, num=2)

        input_flow = tf.placeholder('float', [1, self.height, self.width, 2], name="gru_input_flows")
        
        input_segmentation = tf.placeholder('float', [1, self.height, self.width, self.channels], name="gru_input_unaries")
        
        prev_h = tf.placeholder('float', [1, self.height, self.width, self.channels])
        
        new_h = self.get_GRU_cell(input_images[1], input_images[0], \
             input_flow, prev_h, input_segmentation)

        prediction = tf.argmax(new_h, 3)
        return input_images_tensor, input_flow, input_segmentation, prev_h, new_h, prediction

    def get_GRU_cell(self, input_image, prev_image, flow_input, h_prev, unary_input):
        # apply softmax to h_prev and unary_input
        h_prev = self.softmax_last_dim(h_prev)
        unary_input = self.softmax_last_dim(unary_input)
        h_prev = h_prev - 1./19
        unary_input = unary_input - 1./19

        I_diff = input_image - self.bilinear_warping_module.bilinear_warping(prev_image, flow_input)
        
        # candidate state
        h_prev_warped = self.bilinear_warping_module.bilinear_warping(h_prev, flow_input)

        r = 1. - tf.tanh(tf.abs(tf.nn.conv2d(I_diff, self.weights['ir'], [1,1,1,1], padding='SAME') \
            + self.weights['bias_r']))
        
        h_prev_reset = h_prev_warped * r

        h_tilde = tf.nn.conv2d(unary_input, self.weights['xh'], [1,1,1,1], padding='SAME') \
          + tf.nn.conv2d(h_prev_reset, self.weights['hh'], [1,1,1,1], padding='SAME')

        
        # weighting
        z = tf.sigmoid( \
            tf.nn.conv2d(unary_input, self.weights['xz'], [1,1,1,1], padding='SAME') \
            + tf.nn.conv2d(h_prev_reset, self.weights['hz'], [1,1,1,1], padding='SAME') \
            + self.weights['bias_z']
          )

        h = self.weights['lambda']*(1 - z)*h_prev_reset + z*h_tilde

        return h

    def softmax_last_dim(self, x):
        # apply softmax to a 4D vector along the last dimension
        S = tf.shape(x)
        y = tf.reshape(x, [-1, S[4-1]])
        y = tf.nn.softmax(y)
        y = tf.reshape(y, S)
        return y