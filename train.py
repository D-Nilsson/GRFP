import argparse, glob, os, cv2, sys, pickle, random
import numpy as np
import tensorflow as tf
import config as cfg
from models.stgru import STGRU
from models.lrr import LRR
from models.dilation import dilation10network
from models.flownet2 import Flownet2
from models.flownet1 import Flownet1
from tensorflow.python.framework import ops

bilinear_warping_module = tf.load_op_library('./misc/bilinear_warping.so')
@ops.RegisterGradient("BilinearWarping")
def _BilinearWarping(op, grad):
  return bilinear_warping_module.bilinear_warping_grad(grad, op.inputs[0], op.inputs[1])

class DataLoader():
    def __init__(self, im_size, nbr_frames):
        self.im_size = im_size
        self.dataset_size = [1024, 2048]
        self.nbr_frames = nbr_frames
        self.L = glob.glob(os.path.join(cfg.cityscapes_dir, 'gtFine', 'train', "*", "*labelTrainIds.png"))
        random.shuffle(self.L)
        self.idx = 0

    def get_next_sequence(self):
        H, W = self.dataset_size
        h, w = self.im_size

        offset = [np.random.randint(H - h),
            np.random.randint(W - w)]
        i0, j0 = offset
        i1, j1 = i0 + h, j0 + w

        im_path = self.L[self.idx % len(self.L)]
        self.idx += 1

        parts = im_path.split('/')[-1].split('_')
        city, seq, frame = parts[0], parts[1], parts[2]

        images = []
        gt = cv2.imread(im_path, 0)[i0:i1, j0:j1]
        
        for dt in range(-self.nbr_frames + 1, 1):
            t = int(frame) + dt
            
            frame_path = os.path.join(cfg.cityscapes_video_dir, 'leftImg8bit_sequence', 'train', 
                    city, ("%s_%s_%06d_leftImg8bit.png" % (city, seq, t)))
            images.append(cv2.imread(frame_path, 1).astype(np.float32)[i0:i1,j0:j1][np.newaxis,...])
            
        return images, gt

def train(args):
    nbr_classes = 19

    # learning rates for the GRU and the static segmentation networks, respectively
    learning_rate = 2e-5
    static_learning_rate = 2e-12
    
    # The total number of iterations and when the static network should start being refined
    nbr_iterations = 10000
    t0_dilation_net = 5000

    im_size = [512, 512]
    image_mean = [72.39,82.91,73.16] # the mean is automatically subtracted in some modules e.g. flownet2, so be careful

    f = open('misc/cityscapes_labels.pckl')
    cs_id2trainid, cs_id2name = pickle.load(f)
    f.close()

    assert args.static in ['dilation', 'lrr'], "Only dilation and LRR are supported for now."

    if args.flow == 'flownet2':
        with tf.variable_scope('flow'):
            flow_network = Flownet2(bilinear_warping_module)
            flow_img0 = tf.placeholder(tf.float32)
            flow_img1 = tf.placeholder(tf.float32)
            flow_tensor = flow_network(flow_img0, flow_img1, flip=True)
    elif args.flow == 'flownet1':
        with tf.variable_scope('flow'):
            flow_network = Flownet1()
            flow_img0 = tf.placeholder(tf.float32)
            flow_img1 = tf.placeholder(tf.float32)
            flow_tensor = flow_network.get_output_tensor(flow_img0, flow_img1, im_size)

    RNN = STGRU([nbr_classes, im_size[0], im_size[1]], [7, 7], bilinear_warping_module)
    
    gru_opt, gru_loss, gru_prediction, gru_learning_rate, \
        gru_input_images_tensor, gru_input_flow_tensor, \
        gru_input_segmentation_tensor, gru_targets = RNN.get_optimizer(args.frames)
    unary_grad_op = tf.gradients(gru_loss, gru_input_segmentation_tensor)

    if args.static == 'lrr':
        static_input = tf.placeholder(tf.float32)
        static_network = LRR()
        static_output = static_network(static_input)

        unary_opt, unary_dLdy = static_network.get_optimizer(static_input, static_output, static_learning_rate)
    elif args.static == 'dilation':
        static_input = tf.placeholder(tf.float32)
        static_network = dilation10network()
        static_output = static_network.get_output_tensor(static_input, im_size)

    data_loader = DataLoader(im_size, args.frames)

    loss_history = np.zeros(nbr_iterations)
    loss_history_smoothed = np.zeros(nbr_iterations)

    vars_trainable = [k for k in tf.trainable_variables() if not k.name.startswith('flow/')]
    vars_static = [k for k in vars_trainable if not k in RNN.weights.values()]
    loader_static = tf.train.Saver(vars_static)
    saver = tf.train.Saver(vars_trainable)
    
    if args.flow in ['flownet1', 'flownet2']:
        saver_fn = tf.train.Saver([k for k in tf.trainable_variables() if k.name.startswith('flow/')])

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        if args.static == 'lrr':
            loader_static.restore(sess, './checkpoints/lrr_pretrained')
        elif args.static == 'dilation':
            assert False, "Pretrained dilation model will soon be released."
            saver.restore(sess, './checkpoints/dilation_grfp')

        if args.flow == 'flownet1':
            saver_fn.restore(sess, './checkpoints/flownet1')
        elif args.flow == 'flownet2':
            saver_fn.restore(sess, './checkpoints/flownet2')

        for training_it in range(nbr_iterations):
            images, ground_truth = data_loader.get_next_sequence()

            # Optical flow
            optflow = []
            for frame in range(1, args.frames):
                im, last_im = images[frame], images[frame-1]
                if args.flow == 'flownet2':
                    flow = sess.run(flow_tensor, feed_dict={flow_img0: im, flow_img1: last_im})
                elif args.flow == 'flownet1':
                    flow = sess.run(flow_tensor, feed_dict={flow_img0: im, flow_img1: last_im})
                    flow = flow[...,(1, 0)]
                elif args.flow == 'farneback':
                    im_gray = cv2.cvtColor(im[0], cv2.COLOR_BGR2GRAY)
                    last_im_gray = cv2.cvtColor(last_im[0], cv2.COLOR_BGR2GRAY)

                    flow = cv2.calcOpticalFlowFarneback(im_gray, last_im_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    flow = flow[...,(1, 0)]
                    flow = flow[np.newaxis,...]
                optflow.append(flow)

            # Static segmentation
            static_segm = []
            for frame in range(args.frames):
                im = images[frame]
                if args.static == 'dilation':
                    # augment a 186x186 border around the image and subtract the mean
                    im_aug = cv2.copyMakeBorder(im[0], 186, 186, 186, 186, cv2.BORDER_REFLECT_101)
                    im_aug = im_aug - image_mean
                    im_aug = im_aug[np.newaxis,...]

                    x = sess.run(static_output, feed_dict={static_input: im_aug})
                elif args.static == 'lrr':
                    x = sess.run(static_output, feed_dict={static_input: im})
                static_segm.append(x)

            # GRFP
            rnn_input = {
                gru_learning_rate: learning_rate,
                gru_input_images_tensor: np.stack(images),
                gru_input_flow_tensor: np.stack(optflow),
                gru_input_segmentation_tensor: np.stack(static_segm),
                gru_targets: ground_truth,
            }

            _, loss, pred, unary_grads = sess.run([gru_opt, gru_loss, 
               gru_prediction, unary_grad_op], feed_dict=rnn_input)
            loss_history[training_it] = loss
            
            if training_it < 300:
                loss_history_smoothed[training_it] = np.mean(loss_history[0:training_it+1])
            else:
                loss_history_smoothed[training_it] = 0.997*loss_history_smoothed[training_it-1] + 0.003*loss

            # Refine the static network?
            # The reason that a two-stage training routine is used
            # is because there is not enough GPU memory (with a 12 GB Titan X)
            # to do it in one pass.
            if training_it+1 > t0_dilation_net:
                for k in range(len(images)-3, len(images)):
                    g = unary_grads[0][k]
                    im = images[k]
                    _ = sess.run([unary_opt], feed_dict={
                      static_input: im,
                      unary_dLdy: g
                    })

            if training_it > 0 and (training_it+1) % 1000 == 0:
                saver.save(sess, './checkpoints/%s_%s_it%d' % (args.static, args.flow, training_it+1))

            if (training_it+1) % 200 == 0:
                print("Iteration %d/%d: Loss %.3f" % (training_it+1, nbr_iterations, loss_history_smoothed[training_it]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tran GRFP on the CityScapes training set.')

    parser.add_argument('--static', help='Which static network to use.', required=True)
    parser.add_argument('--flow', help='Which optical flow method to use.', required=True)
    parser.add_argument('--frames', type=int, help='Number of frames to use.', default=5, required=False)

    args = parser.parse_args()

    assert args.flow in ['flownet1', 'flownet2', 'farneback'], "Unknown flow method %s." % args.flow
    assert args.static in ['dilation', 'dilation_grfp', 'lrr', 'lrr_grfp'], "Unknown static method %s." % args.static
    assert args.frames >= 1 and args.frames <= 20, "The number of frames must be between 1 and 20."
    
    train(args)