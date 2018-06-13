import tensorflow as tf

class dilation10network:
    def __init__(self, dropout_keeprate = 1.0):
        #
        self.dropout_keeprate = dropout_keeprate
        self.mean = [72.39,82.91,73.16]

        self.weights = {
            'conv1_1': tf.Variable(tf.zeros([3, 3, 3, 64], dtype=tf.float32), name='conv1_1'),
            'conv1_2': tf.Variable(tf.zeros([3, 3, 64, 64], dtype=tf.float32), name='conv1_2'),
            
            'conv2_1': tf.Variable(tf.zeros([3, 3, 64, 128], dtype=tf.float32), name='conv2_1'),
            'conv2_2': tf.Variable(tf.zeros([3, 3, 128, 128], dtype=tf.float32), name='conv2_2'),
            
            'conv3_1': tf.Variable(tf.zeros([3, 3, 128, 256], dtype=tf.float32), name='conv3_1'),
            'conv3_2': tf.Variable(tf.zeros([3, 3, 256, 256], dtype=tf.float32), name='conv3_2'),
            'conv3_3': tf.Variable(tf.zeros([3, 3, 256, 256], dtype=tf.float32), name='conv3_3'),
            
            'conv4_1': tf.Variable(tf.zeros([3, 3, 256, 512], dtype=tf.float32), name='conv4_1'),
            'conv4_2': tf.Variable(tf.zeros([3, 3, 512, 512], dtype=tf.float32), name='conv4_2'),
            'conv4_3': tf.Variable(tf.zeros([3, 3, 512, 512], dtype=tf.float32), name='conv4_3'),
            
            'conv5_1': tf.Variable(tf.zeros([3, 3, 512, 512], dtype=tf.float32), name='conv5_1'),
            'conv5_2': tf.Variable(tf.zeros([3, 3, 512, 512], dtype=tf.float32), name='conv5_2'),
            'conv5_3': tf.Variable(tf.zeros([3, 3, 512, 512], dtype=tf.float32), name='conv5_3'),
            
            'fc6': tf.Variable(tf.zeros([7, 7, 512, 4096], dtype=tf.float32), name='fc6'),
            'fc7': tf.Variable(tf.zeros([1, 1, 4096, 4096], dtype=tf.float32), name='fc7'),
            'final': tf.Variable(tf.zeros([1, 1, 4096, 19], dtype=tf.float32), name='final'),
            
            'ctx_conv1_1': tf.Variable(tf.zeros([3, 3, 19, 19], dtype=tf.float32), name='ctx_conv1_1'),
            'ctx_conv1_2': tf.Variable(tf.zeros([3, 3, 19, 19], dtype=tf.float32), name='ctx_conv1_1'),
            'ctx_conv2_1': tf.Variable(tf.zeros([3, 3, 19, 19], dtype=tf.float32), name='ctx_conv1_1'),
            'ctx_conv3_1': tf.Variable(tf.zeros([3, 3, 19, 19], dtype=tf.float32), name='ctx_conv1_1'),
            'ctx_conv4_1': tf.Variable(tf.zeros([3, 3, 19, 19], dtype=tf.float32), name='ctx_conv1_1'),
            'ctx_conv5_1': tf.Variable(tf.zeros([3, 3, 19, 19], dtype=tf.float32), name='ctx_conv1_1'),
            'ctx_conv6_1': tf.Variable(tf.zeros([3, 3, 19, 19], dtype=tf.float32), name='ctx_conv1_1'),
            'ctx_conv7_1': tf.Variable(tf.zeros([3, 3, 19, 19], dtype=tf.float32), name='ctx_conv1_1'),
            'ctx_fc1': tf.Variable(tf.zeros([3, 3, 19, 19], dtype=tf.float32), name='ctx_conv1_1'),
            'ctx_final': tf.Variable(tf.zeros([1, 1, 19, 19], dtype=tf.float32), name='ctx_conv1_1'),
            'ctx_upsample': tf.Variable(tf.zeros([16, 16, 19, 19], dtype=tf.float32), name='ctx_conv1_1'),
        }
        self.biases = {
            'conv1_1': tf.Variable(tf.zeros([64], dtype=tf.float32), name='conv1_1_b'),
            'conv1_2': tf.Variable(tf.zeros([64], dtype=tf.float32), name='conv1_2_b'),
            
            'conv2_1': tf.Variable(tf.zeros([128], dtype=tf.float32), name='conv2_1_b'),
            'conv2_2': tf.Variable(tf.zeros([128], dtype=tf.float32), name='conv2_2_b'),
            
            'conv3_1': tf.Variable(tf.zeros([256], dtype=tf.float32), name='conv3_1_b'),
            'conv3_2': tf.Variable(tf.zeros([256], dtype=tf.float32), name='conv3_2_b'),
            'conv3_3': tf.Variable(tf.zeros([256], dtype=tf.float32), name='conv3_3_b'),
            
            'conv4_1': tf.Variable(tf.zeros([512], dtype=tf.float32), name='conv4_1_b'),
            'conv4_2': tf.Variable(tf.zeros([512], dtype=tf.float32), name='conv4_2_b'),
            'conv4_3': tf.Variable(tf.zeros([512], dtype=tf.float32), name='conv4_3_b'),
            
            'conv5_1': tf.Variable(tf.zeros([512], dtype=tf.float32), name='conv5_1_b'),
            'conv5_2': tf.Variable(tf.zeros([512], dtype=tf.float32), name='conv5_2_b'),
            'conv5_3': tf.Variable(tf.zeros([512], dtype=tf.float32), name='conv5_3_b'),
            
            'fc6': tf.Variable(tf.zeros([4096], dtype=tf.float32), name='fc6_b'),
            'fc7': tf.Variable(tf.zeros([4096], dtype=tf.float32), name='fc7_b'),
            'final': tf.Variable(tf.zeros([19], dtype=tf.float32), name='final_b'),
            
            'ctx_conv1_1': tf.Variable(tf.zeros([19], dtype=tf.float32), name='ctx_conv1_1_b'),
            'ctx_conv1_2': tf.Variable(tf.zeros([19], dtype=tf.float32), name='ctx_conv1_1_b'),
            'ctx_conv2_1': tf.Variable(tf.zeros([19], dtype=tf.float32), name='ctx_conv1_1_b'),
            'ctx_conv3_1': tf.Variable(tf.zeros([19], dtype=tf.float32), name='ctx_conv1_1_b'),
            'ctx_conv4_1': tf.Variable(tf.zeros([19], dtype=tf.float32), name='ctx_conv1_1_b'),
            'ctx_conv5_1': tf.Variable(tf.zeros([19], dtype=tf.float32), name='ctx_conv1_1_b'),
            'ctx_conv6_1': tf.Variable(tf.zeros([19], dtype=tf.float32), name='ctx_conv1_1_b'),
            'ctx_conv7_1': tf.Variable(tf.zeros([19], dtype=tf.float32), name='ctx_conv1_1_b'),
            'ctx_fc1': tf.Variable(tf.zeros([19], dtype=tf.float32), name='ctx_conv1_1_b'),
            'ctx_final': tf.Variable(tf.zeros([19], dtype=tf.float32), name='ctx_conv1_1_b'),
        }

    def get_output_tensor(self, x, out_size):
        # returns output tensor
        output_shape = [1, out_size[0], out_size[1], 19]

        conv1_1 = tf.nn.relu(tf.nn.conv2d(x, self.weights['conv1_1'], strides=[1,1,1,1], padding="VALID") + self.biases['conv1_1'])
        conv1_2 = tf.nn.relu(tf.nn.conv2d(conv1_1, self.weights['conv1_2'], strides=[1,1,1,1], padding="VALID") + self.biases['conv1_2'])
        conv1_2 = tf.nn.max_pool(conv1_2, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        conv2_1 = tf.nn.relu(tf.nn.conv2d(conv1_2, self.weights['conv2_1'], strides=[1,1,1,1], padding="VALID") + self.biases['conv2_1'])
        conv2_2 = tf.nn.relu(tf.nn.conv2d(conv2_1, self.weights['conv2_2'], strides=[1,1,1,1], padding="VALID") + self.biases['conv2_2'])
        conv2_2 = tf.nn.max_pool(conv2_2, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        conv3_1 = tf.nn.relu(tf.nn.conv2d(conv2_2, self.weights['conv3_1'], strides=[1,1,1,1], padding="VALID") + self.biases['conv3_1'])
        conv3_2 = tf.nn.relu(tf.nn.conv2d(conv3_1, self.weights['conv3_2'], strides=[1,1,1,1], padding="VALID") + self.biases['conv3_2'])
        conv3_3 = tf.nn.relu(tf.nn.conv2d(conv3_2, self.weights['conv3_3'], strides=[1,1,1,1], padding="VALID") + self.biases['conv3_3'])
        conv3_3 = tf.nn.max_pool(conv3_3, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        conv4_1 = tf.nn.relu(tf.nn.conv2d(conv3_3, self.weights['conv4_1'], strides=[1,1,1,1], padding="VALID") + self.biases['conv4_1'])
        conv4_2 = tf.nn.relu(tf.nn.conv2d(conv4_1, self.weights['conv4_2'], strides=[1,1,1,1], padding="VALID") + self.biases['conv4_2'])
        conv4_3 = tf.nn.relu(tf.nn.conv2d(conv4_2, self.weights['conv4_3'], strides=[1,1,1,1], padding="VALID") + self.biases['conv4_3'])
        # not pooling, instead dilations in the folling ops

        conv5_1 = tf.nn.relu(tf.nn.atrous_conv2d(conv4_3, self.weights['conv5_1'], padding="VALID", rate=2) + self.biases['conv5_1'])
        conv5_2 = tf.nn.relu(tf.nn.atrous_conv2d(conv5_1, self.weights['conv5_2'], padding="VALID", rate=2) + self.biases['conv5_2'])
        conv5_3 = tf.nn.relu(tf.nn.atrous_conv2d(conv5_2, self.weights['conv5_3'], padding="VALID", rate=2) + self.biases['conv5_3'])

        fc6 = tf.nn.relu(tf.nn.atrous_conv2d(conv5_3, self.weights['fc6'], padding="VALID", rate=4) + self.biases['fc6'])
        fc6 = tf.nn.dropout(fc6, self.dropout_keeprate)
        fc7 = tf.nn.relu(tf.nn.atrous_conv2d(fc6, self.weights['fc7'], padding="VALID", rate=4) + self.biases['fc7'])
        fc7 = tf.nn.dropout(fc7, self.dropout_keeprate)
        final = tf.nn.atrous_conv2d(fc7, self.weights['final'], padding="VALID", rate=4) + self.biases['final']

        ctx_conv1_1 = tf.nn.relu(tf.nn.conv2d(final, self.weights['ctx_conv1_1'], strides=[1,1,1,1], padding="SAME") + self.biases['ctx_conv1_1'])
        ctx_conv1_2 = tf.nn.relu(tf.nn.conv2d(ctx_conv1_1, self.weights['ctx_conv1_2'], strides=[1,1,1,1], padding="SAME") + self.biases['ctx_conv1_2'])
        ctx_conv2_1 = tf.nn.relu(tf.nn.atrous_conv2d(ctx_conv1_2, self.weights['ctx_conv2_1'], padding="SAME", rate=2) + self.biases['ctx_conv2_1'])
        ctx_conv3_1 = tf.nn.relu(tf.nn.atrous_conv2d(ctx_conv2_1, self.weights['ctx_conv3_1'], padding="SAME", rate=4) + self.biases['ctx_conv3_1'])
        ctx_conv4_1 = tf.nn.relu(tf.nn.atrous_conv2d(ctx_conv3_1, self.weights['ctx_conv4_1'], padding="SAME", rate=8) + self.biases['ctx_conv4_1'])
        ctx_conv5_1 = tf.nn.relu(tf.nn.atrous_conv2d(ctx_conv4_1, self.weights['ctx_conv5_1'], padding="SAME", rate=16) + self.biases['ctx_conv5_1'])
        ctx_conv6_1 = tf.nn.relu(tf.nn.atrous_conv2d(ctx_conv5_1, self.weights['ctx_conv6_1'], padding="SAME", rate=32) + self.biases['ctx_conv6_1'])
        ctx_conv7_1 = tf.nn.relu(tf.nn.atrous_conv2d(ctx_conv6_1, self.weights['ctx_conv7_1'], padding="SAME", rate=64) + self.biases['ctx_conv7_1'])
        
        ctx_fc1 = tf.nn.relu(tf.nn.conv2d(ctx_conv7_1, self.weights['ctx_fc1'], strides=[1,1,1,1], padding="SAME") + self.biases['ctx_fc1'])
        ctx_final = tf.nn.conv2d(ctx_fc1, self.weights['ctx_final'], strides=[1,1,1,1], padding="SAME") + self.biases['ctx_final']
        ctx_upsample = tf.nn.conv2d_transpose(ctx_final, self.weights['ctx_upsample'], output_shape=output_shape, strides=[1,8,8,1])

        return ctx_upsample

    def get_optimizer(self, x, y, learning_rate):
        # optimize wrt the ctx_* variables
        dLdy = tf.placeholder('float')

        # the correct values will backpropagate to ctx_upsample
        loss = tf.reduce_sum(dLdy * y)

        #opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.95, beta2=0.99, epsilon=1e-8)
        opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        opt = opt.minimize(loss,
            [v for k, v in self.weights.iteritems() if (k[0:4] == 'ctx_' or k in ['fc6', 'fc7', 'final'])].extend(
            [v for k, v in self.biases.iteritems() if (k[0:4] == 'ctx_' or k in ['fc6', 'fc7', 'final'])]))

        return opt, dLdy