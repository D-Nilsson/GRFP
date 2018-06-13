import argparse, glob, os, cv2, sys, pickle
import numpy as np
import tensorflow as tf
import config as cfg
from models.stgru import STGRU
from models.lrr import LRR
from models.flownet2 import Flownet2
from models.flownet1 import Flownet1
from tensorflow.python.framework import ops

sys.path.insert(0, os.path.join(cfg.cityscapes_scripts_root, 'evaluation'))
import evalPixelLevelSemanticLabeling

bilinear_warping_module = tf.load_op_library('./misc/bilinear_warping.so')
@ops.RegisterGradient("BilinearWarping")
def _BilinearWarping(op, grad):
  return bilinear_warping_module.bilinear_warping_grad(grad, op.inputs[0], op.inputs[1])

def evaluate(args):
    data_split = 'val'
    nbr_classes = 19
    im_size = [1024, 2048]
    image_mean = [72.39,82.91,73.16] # the mean is automatically subtracted in some modules e.g. flownet2, so be careful

    f = open('misc/cityscapes_labels.pckl')
    cs_id2trainid, cs_id2name = pickle.load(f)
    f.close()

    assert args.static == 'lrr', "Only LRR is supported for now."
    
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
    
    input_images_tensor, input_flow, \
        input_segmentation, prev_h, new_h, \
        prediction = RNN.get_one_step_predictor()

    static_input = tf.placeholder(tf.float32)
    static_network = LRR()
    static_output = static_network(static_input)

    saver = tf.train.Saver([k for k in tf.global_variables() if not k.name.startswith('flow/')])
    if args.flow in ['flownet1', 'flownet2']:
        saver_fn = tf.train.Saver([k for k in tf.global_variables() if k.name.startswith('flow/')])

    with tf.Session() as sess:
        saver.restore(sess, './checkpoints/lrr_grfp')
        
        if args.flow == 'flownet1':
            saver_fn.restore(sess, './checkpoints/flownet1')
        elif args.flow == 'flownet2':
            saver_fn.restore(sess, './checkpoints/flownet2')

        L = glob.glob(os.path.join(cfg.cityscapes_dir, 'gtFine', data_split, "*", "*labelIds.png"))
        for (progress_counter, im_path) in enumerate(L):
            parts = im_path.split('/')[-1].split('_')
            city, seq, frame = parts[0], parts[1], parts[2]

            print("Processing sequence %d/%d" % (progress_counter+1, len(L)))
            for dt in range(-args.frames + 1, 1):
                first_frame = dt == -args.frames + 1
                t = int(frame) + dt
                
                frame_path = os.path.join(cfg.cityscapes_video_dir, 'leftImg8bit_sequence', data_split, 
                        city, ("%s_%s_%06d_leftImg8bit.png" % (city, seq, t)))
                im = cv2.imread(frame_path, 1).astype(np.float32)[np.newaxis,...]

                # Compute optical flow
                if not first_frame:
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

                # Static segmentation
                x = sess.run(static_output, feed_dict={static_input: im})
                
                if first_frame:
                    # the hidden state is simple the static segmentation for the first frame
                    h = x
                    pred = np.argmax(h, axis=3)
                else:
                    inputs = {
                        input_images_tensor: np.stack([last_im, im]),
                        input_flow: flow,
                        input_segmentation: x,
                        prev_h: h
                    }
                    # GRFP
                    h, pred = sess.run([new_h, prediction], feed_dict=inputs)

                last_im = im

            # save it
            S = pred[0]
            S_new = S.copy()
            for (idx, train_idx) in cs_id2trainid.iteritems():
                S_new[S == train_idx] = idx

            output_path = '%s_%s_%s.png' % (city, seq, frame)
            cv2.imwrite(os.path.join(cfg.cityscapes_dir, 'results', output_path), S_new)


        # Evaluate using the official CityScapes code
        evalPixelLevelSemanticLabeling.main([])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evluate GRFP on the CityScapes validation set.')

    parser.add_argument('--static', help='Which static network to use.', required=True)
    parser.add_argument('--flow', help='Which optical flow method to use.', required=True)
    parser.add_argument('--frames', type=int, help='Number of frames to use.', default=5, required=False)

    args = parser.parse_args()

    assert args.flow in ['flownet1', 'flownet2', 'farneback'], "Unknown flow method %s." % args.flow
    assert args.static in ['dilation', 'dilation_grfp', 'lrr', 'lrr_grfp'], "Unknown static method %s." % args.static
    assert args.frames >= 1 and args.frames <= 20, "The number of frames must be between 1 and 20."
    
    evaluate(args)