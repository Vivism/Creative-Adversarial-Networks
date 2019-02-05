import os
import scipy.misc
from glob import glob

from model import DCGAN
from utils import pp, visualize, show_all_variables

import tensorflow as tf
from slim.nets import nets_factory

from libs import aws

from flags import FLAGS

def main(_):
    print('\nProcessing Flags\n')
    pp.pprint(FLAGS.__flags)
  
    # check to see if we want to use AWS S3
    if FLAGS.use_s3:
        if FLAGS.s3_bucket is None:
            raise ValueError('use_s3 flag set, but no bucket set. ')
        # check to see if s3 bucket exists:
        elif not aws.bucket_exists(FLAGS.s3_bucket):
            raise ValueError('`use_s3` flag set, but bucket "%s" doesn\'t exist. Not using s3' % FLAGS.s3_bucket)

    # set dimensions
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    # configure the log_dir to match the params
    dir_string = "isCan={},lr={},imsize={},hasStyleNet={},batch_size={}".format(
        FLAGS.can,
        FLAGS.learning_rate,
        FLAGS.input_height,
        FLAGS.style_net_checkpoint is not None,
        FLAGS.batch_size
    )
    log_dir = os.path.join(FLAGS.log_dir, dir_string)

    # Set log directory
    if not glob(log_dir + "*"):
        log_dir = os.path.join(log_dir, "000")
    else:
        containing_dir = os.path.join(log_dir, "[0-9][0-9][0-9]")
        nums = [int(x[-3:]) for x in glob(containing_dir)] # TODO FIX THESE HACKS
        if nums == []:
            num = 0
        else:
            num = max(nums) + 1
        log_dir = os.path.join(log_dir,"{:03d}".format(num))
    FLAGS.log_dir = log_dir

    # set checkpoint directory
    if FLAGS.checkpoint_dir is None:
        FLAGS.checkpoint_dir = os.path.join(FLAGS.log_dir, 'checkpoint')
        FLAGS.use_default_checkpoint = True
    elif FLAGS.use_default_checkpoint:
        raise ValueError('`use_default_checkpoint` flag only works if you keep checkpoint_dir as None')

    if FLAGS.sample_dir is None:
        FLAGS.sample_dir = os.path.join(FLAGS.log_dir, 'samples')

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    print('\nFlags Processed\n')

    if FLAGS.style_net_checkpoint:
        network_fn = nets_factory

    if FLAGS.train and FLAGS.sample_data_dir is None:
        raise ValueError('Sample Data Directory required during training')

    # setup DCGAN
    sess = None

    dcgan = DCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.sample_size,
        use_resize=FLAGS.use_resize,
        replay=FLAGS.replay,
        y_dim=27, # default for wikiart dataset
        smoothing=FLAGS.smoothing,
        lamb = FLAGS.lambda_val,
        input_fname_pattern=FLAGS.input_fname_pattern,
        crop=FLAGS.crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        wgan=FLAGS.wgan,
        learning_rate = FLAGS.learning_rate,
        style_net_checkpoint=FLAGS.style_net_checkpoint,
        can=FLAGS.can,
        sample_data_dir=FLAGS.sample_data_dir
    )

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=FLAGS.allow_gpu_growth
    with tf.Session(config=run_config) as sess:
        dcgan.set_sess(sess)
        # show_all_variables()

        if FLAGS.train:
            dcgan.train(FLAGS)
        # else:
        #   if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        #     raise Exception("[!] Train a model first, then run test mode")

        visualize(sess, dcgan, FLAGS)

if __name__ == '__main__':
    tf.app.run()
