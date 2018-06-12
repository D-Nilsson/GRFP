import glob
import numpy as np
import tensorflow as tf
import scipy.io

class LRR:
    def __init__(self):
        self.weights = dict()

        for key, shape in self.all_variables():
            self.weights[key] = tf.get_variable(key, shape=shape)

    def __call__(self, im):
        return self.get_blobs(im)['prediction_4x']
    
    def get_optimizer(self, x, y, learning_rate):
        dLdy = tf.placeholder('float')

        # the correct values will backpropagate to y
        loss = tf.reduce_sum(dLdy * y)

        opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        opt = opt.minimize(loss,
            var_list=[v for k, v in self.weights.iteritems() if not ('_bil_' in k or 'deconv' in k or 'bases' in k)])

        return opt, dLdy
    
    def get_blobs(self, im, dropout_keeprate=1):
        blobs = dict()

        # to rgb and mean subtraction
        # it expects a bgr image without the mean subtracted
        #im = im[:,:,:,(2,1,0)]
        im = tf.concat(tf.split(im, 3, axis=3)[::-1], axis=3)
        im = im - np.array([73.1652, 82.9206, 72.4080])

        batch_size = tf.to_int32(tf.shape(im)[0])
        width = tf.to_int32(tf.shape(im)[2])
        height = tf.to_int32(tf.shape(im)[1])
        TARGET_WIDTH = width
        TARGET_HEIGHT = height

        divisor = 32.
        ADAPTED_WIDTH = tf.to_int32(tf.ceil(tf.to_float(width)/divisor) * divisor)
        ADAPTED_HEIGHT = tf.to_int32(tf.ceil(tf.to_float(height)/divisor) * divisor)

        SCALE_WIDTH = tf.to_float(width) / tf.to_float(ADAPTED_WIDTH);
        SCALE_HEIGHT = tf.to_float(height) / tf.to_float(ADAPTED_HEIGHT);

        blobs['x1'] = tf.nn.conv2d(im, self.weights['conv1_1f'], [1,1,1,1], padding='SAME') + self.weights['conv1_1b']
        blobs['x2'] = tf.nn.relu(blobs['x1'])
        blobs['x3'] = tf.nn.conv2d(blobs['x2'], self.weights['conv1_2f'], [1,1,1,1], padding='SAME') + self.weights['conv1_2b']
        blobs['x4'] = tf.nn.relu(blobs['x3'])
        blobs['x5'] = tf.nn.max_pool(blobs['x4'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        blobs['x6'] = tf.nn.conv2d(blobs['x5'], self.weights['conv2_1f'], [1,1,1,1], padding='SAME') + self.weights['conv2_1b']
        blobs['x7'] = tf.nn.relu(blobs['x6'])
        blobs['x8'] = tf.nn.conv2d(blobs['x7'], self.weights['conv2_2f'], [1,1,1,1], padding='SAME') + self.weights['conv2_2b']
        blobs['x9'] = tf.nn.relu(blobs['x8'])
        blobs['x10'] = tf.nn.max_pool(blobs['x9'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        blobs['x11'] = tf.nn.conv2d(blobs['x10'], self.weights['conv3_1f'], [1,1,1,1], padding='SAME') + self.weights['conv3_1b']
        blobs['x12'] = tf.nn.relu(blobs['x11'])
        blobs['x13'] = tf.nn.conv2d(blobs['x12'], self.weights['conv3_2f'], [1,1,1,1], padding='SAME') + self.weights['conv3_2b']
        blobs['x14'] = tf.nn.relu(blobs['x13'])
        blobs['x15'] = tf.nn.conv2d(blobs['x14'], self.weights['conv3_3f'], [1,1,1,1], padding='SAME') + self.weights['conv3_3b']
        blobs['x16'] = tf.nn.relu(blobs['x15'])
        blobs['x17'] = tf.nn.max_pool(blobs['x16'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        
        blobs['x18'] = tf.nn.conv2d(blobs['x17'], self.weights['conv4_1f'], [1,1,1,1], padding='SAME') + self.weights['conv4_1b']
        blobs['x19'] = tf.nn.relu(blobs['x18'])
        blobs['x20'] = tf.nn.conv2d(blobs['x19'], self.weights['conv4_2f'], [1,1,1,1], padding='SAME') + self.weights['conv4_2b']
        blobs['x21'] = tf.nn.relu(blobs['x20'])
        blobs['x22'] = tf.nn.conv2d(blobs['x21'], self.weights['conv4_3f'], [1,1,1,1], padding='SAME') + self.weights['conv4_3b']
        blobs['x23'] = tf.nn.relu(blobs['x22'])
        blobs['x24'] = tf.nn.max_pool(blobs['x23'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        blobs['x25'] = tf.nn.conv2d(blobs['x24'], self.weights['conv5_1f'], [1,1,1,1], padding='SAME') + self.weights['conv5_1b']
        blobs['x26'] = tf.nn.relu(blobs['x25'])
        blobs['x27'] = tf.nn.conv2d(blobs['x26'], self.weights['conv5_2f'], [1,1,1,1], padding='SAME') + self.weights['conv5_2b']
        blobs['x28'] = tf.nn.relu(blobs['x27'])
        blobs['x29'] = tf.nn.conv2d(blobs['x28'], self.weights['conv5_3f'], [1,1,1,1], padding='SAME') + self.weights['conv5_3b']
        blobs['x30'] = tf.nn.relu(blobs['x29'])
        blobs['x31'] = tf.nn.max_pool(blobs['x30'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        blobs['x32'] = tf.nn.conv2d(blobs['x31'], self.weights['fc6f'], [1,1,1,1], padding='SAME') + self.weights['fc6b']
        blobs['x33'] = tf.nn.relu(blobs['x32'])
        blobs['x34'] = tf.nn.dropout(blobs['x33'], keep_prob=dropout_keeprate)

        blobs['x35'] = tf.nn.conv2d(blobs['x34'], self.weights['fc7f'], [1,1,1,1], padding='SAME') + self.weights['fc7b']
        blobs['x36'] = tf.nn.relu(blobs['x35'])
        blobs['x37'] = tf.nn.dropout(blobs['x36'], keep_prob=dropout_keeprate)

        ### end of VGG

        blobs['coef_32x'] = tf.nn.conv2d(blobs['x37'], self.weights['bases_coef_32xf'], [1,1,1,1], padding='SAME') + self.weights['bases_coef_32xb']

        # groups parameter for a deconv layer is not supported in tensorflow, hacky solution below
        blobs['prediction_32x'] = tf.concat([tf.nn.conv2d_transpose(tf.slice(blobs['coef_32x'], [0, 0, 0, 10*k], [-1, -1, -1, 10]), tf.slice(self.weights['deconv_32xf'], [0, 0, 0, 10*k], [-1, -1, -1, 10]), output_shape=[batch_size, ADAPTED_HEIGHT/8, ADAPTED_WIDTH/8, 1], strides=[1,4,4,1]) for k in range(19)], axis=3)
        

        blobs['dil_seg32x_coef'] = tf.nn.conv2d(blobs['x37'], self.weights['dil_seg32x_coeff'], [1,1,1,1], padding='SAME') + self.weights['dil_seg32x_coefb']

        blobs['dil_seg32x'] = tf.concat([tf.nn.conv2d_transpose(tf.slice(blobs['dil_seg32x_coef'], [0, 0, 0, 10*k], [-1, -1, -1, 10]), tf.slice(self.weights['dil_seg_deconv_32xf'], [0, 0, 0, 10*k], [-1, -1, -1, 10]), output_shape=[batch_size, ADAPTED_HEIGHT/8, ADAPTED_WIDTH/8, 1], strides=[1,4,4,1]) for k in range(19)], axis=3) + self.weights['dil_mask_deconv32sb']

        blobs['ero_seg32x_coef'] = tf.nn.conv2d(blobs['x37'], self.weights['ero_mask32s_coeff'], [1,1,1,1], padding='SAME') + self.weights['ero_seg32x_coefb']

        blobs['ero_seg32x'] = tf.concat([tf.nn.conv2d_transpose(tf.slice(blobs['ero_seg32x_coef'], [0, 0, 0, 10*k], [-1, -1, -1, 10]), tf.slice(self.weights['ero_seg_deconv_32xf'], [0, 0, 0, 10*k], [-1, -1, -1, 10]), output_shape=[batch_size, ADAPTED_HEIGHT/8, ADAPTED_WIDTH/8, 1], strides=[1,4,4,1]) for k in range(19)], axis=3) + self.weights['ero_mask_deconv32sb']

        blobs['coef_16x'] = tf.nn.conv2d(blobs['x30'], self.weights['bases_coef_16sf'], [1,1,1,1], padding='SAME') + self.weights['bases_coef_16xb']

        blobs['prediction_16x_add'] = tf.concat([tf.nn.conv2d_transpose(tf.slice(blobs['coef_16x'], [0, 0, 0, 10*k], [-1, -1, -1, 10]), tf.slice(self.weights['deconv_16xf'], [0, 0, 0, 10*k], [-1, -1, -1, 10]), output_shape=[batch_size, ADAPTED_HEIGHT/4, ADAPTED_WIDTH/4, 1], strides=[1,4,4,1]) for k in range(19)], axis=3)

        
        blobs['prediction_32x_bil_x2'] = tf.concat([tf.nn.conv2d_transpose(tf.slice(blobs['prediction_32x'], [0, 0, 0, k], [-1, -1, -1, 1]), tf.slice(self.weights['dec_prediction_32x_bil_x2f'], [0, 0, 0, k], [-1, -1, -1, 1]), output_shape=[batch_size, ADAPTED_HEIGHT/4, ADAPTED_WIDTH/4, 1], strides=[1,2,2,1]) for k in range(19)], axis=3)


        blobs['prob_32x'] = tf.nn.softmax(blobs['prediction_32x_bil_x2'])

        blobs['prob_32x_dilate'] = tf.nn.max_pool(blobs['prob_32x'], [1, 17, 17, 1], [1,1,1,1], padding='SAME')

        blobs['neg_prob_32x'] = -blobs['prob_32x']

        blobs['neg_prob_32x_dilate'] = tf.nn.max_pool(blobs['neg_prob_32x'], [1, 17, 17, 1], [1,1,1,1], padding='SAME')

        blobs['bound_mask32x'] = blobs['prob_32x_dilate'] + blobs['neg_prob_32x_dilate']

        blobs['pred_16x_aft_DP'] = blobs['bound_mask32x'] * blobs['prediction_16x_add']

        blobs['coef_8x'] = tf.nn.conv2d(blobs['x23'], self.weights['bases_coef_8xf'], [1,1,1,1], padding='SAME') + self.weights['bases_coef_8xb']

        blobs['prediction_8x_add'] = tf.concat([tf.nn.conv2d_transpose(tf.slice(blobs['coef_8x'], [0, 0, 0, 10*k], [-1, -1, -1, 10]), tf.slice(self.weights['deconv_8xf'], [0, 0, 0, 10*k], [-1, -1, -1, 10]), output_shape=[batch_size, ADAPTED_HEIGHT/2, ADAPTED_WIDTH/2, 1], strides=[1,4,4,1]) for k in range(19)], axis=3)

        blobs['prediction_16x'] = blobs['prediction_32x_bil_x2'] + blobs['pred_16x_aft_DP']

        blobs['prediction_16x_bil_x2'] = tf.concat([tf.nn.conv2d_transpose(tf.slice(blobs['prediction_16x'], [0, 0, 0, k], [-1, -1, -1, 1]), tf.slice(self.weights['dec_prediction_16x_bil_x2f'], [0, 0, 0, k], [-1, -1, -1, 1]), output_shape=[batch_size, ADAPTED_HEIGHT/2, ADAPTED_WIDTH/2, 1], strides=[1,2,2,1]) for k in range(19)], axis=3)


        blobs['prob_16x'] = tf.nn.softmax(blobs['prediction_16x_bil_x2'])

        blobs['prob_16x_dilate'] = tf.nn.max_pool(blobs['prob_16x'], [1, 17, 17, 1], [1,1,1,1], padding='SAME')

        blobs['neg_prob_16x'] = -blobs['prob_16x']

        blobs['neg_prob_16x_dilate'] = tf.nn.max_pool(blobs['neg_prob_16x'], [1, 17, 17, 1], [1,1,1,1], padding='SAME')

        blobs['bound_mask16x'] = blobs['prob_16x_dilate'] + blobs['neg_prob_16x_dilate']

        blobs['pred_8x_aft_DP'] = blobs['bound_mask16x'] * blobs['prediction_8x_add']

        blobs['prediction_8x'] = blobs['prediction_16x_bil_x2'] + blobs['pred_8x_aft_DP']

        blobs['coef_4x'] = tf.nn.conv2d(blobs['x16'], self.weights['bases_coef_4xf'], [1,1,1,1], padding='SAME') + self.weights['bases_coef_4xb']

        blobs['prediction_4x_add'] = tf.concat([tf.nn.conv2d_transpose(tf.slice(blobs['coef_4x'], [0, 0, 0, 10*k], [-1, -1, -1, 10]), tf.slice(self.weights['deconv_4xf'], [0, 0, 0, 10*k], [-1, -1, -1, 10]), output_shape=[batch_size, ADAPTED_HEIGHT, ADAPTED_WIDTH, 1], strides=[1,4,4,1]) for k in range(19)], axis=3)


        blobs['prediction_8x_bil_x2'] = tf.concat([tf.nn.conv2d_transpose(tf.slice(blobs['prediction_8x'], [0, 0, 0, k], [-1, -1, -1, 1]), tf.slice(self.weights['dec_prediction_8x_bil_x2f'], [0, 0, 0, k], [-1, -1, -1, 1]), output_shape=[batch_size, ADAPTED_HEIGHT, ADAPTED_WIDTH, 1], strides=[1,2,2,1]) for k in range(19)], axis=3)


        blobs['prob_8x'] = tf.nn.softmax(blobs['prediction_8x_bil_x2'])

        blobs['prob_8x_dilate'] = tf.nn.max_pool(blobs['prob_8x'], [1, 17, 17, 1], [1,1,1,1], padding='SAME')

        blobs['neg_prob_8x'] = -blobs['prob_8x']

        blobs['neg_prob_8x_dilate'] = tf.nn.max_pool(blobs['neg_prob_8x'], [1, 17, 17, 1], [1,1,1,1], padding='SAME')

        blobs['bound_mask8x'] = blobs['prob_8x_dilate'] + blobs['neg_prob_8x_dilate']

        blobs['pred_4x_aft_DP'] = blobs['bound_mask8x'] * blobs['prediction_4x_add']

        blobs['prediction_4x'] = blobs['prediction_8x_bil_x2'] + blobs['pred_4x_aft_DP']
        ############# DONE ###############

        return blobs

    # Hacky, but it works. I used to load from numpy files in __init__
    def all_variables(self):
        return [('deconv_16xf', (8, 8, 1, 190)),
            ('conv3_1b', (256,)), 
            ('bases_coef_32xb', (190,)),
            ('dil_seg32x_coeff', (5, 5, 4096, 190)),
            ('conv3_2f', (3, 3, 256, 256)),
            ('fc6f', (7, 7, 512, 4096)),
            ('conv5_2b', (512,)), 
            ('conv4_2f', (3, 3, 512, 512)),
            ('conv5_3f', (3, 3, 512, 512)),
            ('conv4_3f', (3, 3, 512, 512)),
            ('bases_coef_16xb', (190,)),
            ('dil_seg32x_coefb', (190,)),
            ('deconv_32xf', (8, 8, 1, 190)),
            ('bases_coef_16sf', (5, 5, 512, 190)),
            ('ero_mask32s_coeff', (5, 5, 4096, 190)),
            ('conv4_1b', (512,)), 
            ('bases_coef_32xf', (5, 5, 4096, 190)),
            ('conv2_2b', (128,)), 
            ('conv1_2b', (64,)),  
            ('conv3_3f', (3, 3, 256, 256)),
            ('dec_prediction_8x_bil_x2f', (4, 4, 1, 19)),
            ('conv3_2b', (256,)), 
            ('bases_coef_8xb', (190,)),
            ('fc7f', (1, 1, 4096, 4096)),
            ('ero_seg_deconv_32xf', (8, 8, 1, 190)),
            ('conv2_1b', (128,)), 
            ('conv1_2f', (3, 3, 64, 64)),
            ('fc6b', (4096,)),
            ('conv4_2b', (512,)), 
            ('ero_seg32x_coefb', (190,)),
            ('bases_coef_8xf', (5, 5, 512, 190)),
            ('conv5_2f', (3, 3, 512, 512)),
            ('conv1_1b', (64,)),  
            ('conv5_1f', (3, 3, 512, 512)),
            ('bases_coef_4xb', (190,)),
            ('conv3_1f', (3, 3, 128, 256)),
            ('dec_prediction_16x_bil_x2f', (4, 4, 1, 19)),
            ('conv3_3b', (256,)), 
            ('conv2_1f', (3, 3, 64, 128)),
            ('deconv_4xf', (8, 8, 1, 190)),
            ('dil_mask_deconv32sb', (19,)),
            ('conv5_3b', (512,)), 
            ('fc7b', (4096,)),
            ('conv2_2f', (3, 3, 128, 128)),
            ('conv4_3b', (512,)), 
            ('conv5_1b', (512,)), 
            ('bases_coef_4xf', (5, 5, 256, 190)),
            ('conv4_1f', (3, 3, 256, 512)),
            ('dec_prediction_32x_bil_x2f', (4, 4, 1, 19)),
            ('deconv_8xf', (8, 8, 1, 190)),
            ('conv1_1f', (3, 3, 3, 64)),
            ('dil_seg_deconv_32xf', (8, 8, 1, 190)),
            ('ero_mask_deconv32sb', (19,))]