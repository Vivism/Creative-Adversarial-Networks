from __future__ import division
import re
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from random import shuffle

from slim.nets import nets_factory
import generators
import discriminators

from ops import *
from utils import *
from losses import *

from libs import aws

class DCGAN(object):
  def __init__(
    self,
    sess,
    input_height=108,
    input_width=108,
    crop=True,
    batch_size=64,
    sample_num = 64,
    output_height=64,
    output_width=64,
    y_dim=None,
    z_dim=100,
    gf_dim=64,
    df_dim=32,
    smoothing=0.9,
    lamb = 1.0,
    use_resize=False,
    replay=False,
    learning_rate = 1e-4,
    style_net_checkpoint=None,
    gfc_dim=1024,
    dfc_dim=1024,
    c_dim=3,
    wgan=False,
    can=True,
    input_fname_pattern='*.jpg',
    checkpoint_dir=None,
    sample_dir=None,
    old_model=False,
    sample_data_dir=''
  ):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width
    self.learning_rate = learning_rate


    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn0 = batch_norm(name='d_bn0')
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')
    self.d_bn4 = batch_norm(name='d_bn4')
    self.d_bn5 = batch_norm(name='d_bn5')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')
    self.g_bn4 = batch_norm(name='g_bn4')
    self.g_bn5 = batch_norm(name='g_bn5')

    # variables that determines whether to use style net separate from discriminator
    self.style_net_checkpoint = style_net_checkpoint

    self.smoothing = smoothing
    self.lamb = lamb

    self.can = can
    self.wgan = wgan
    self.use_resize = use_resize
    self.replay = replay
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.experience_flag = False

    self.sample_data_dir = os.path.expanduser(sample_data_dir)

    # Dataset is Wikiart
    data_path = os.path.join(self.sample_data_dir, self.input_fname_pattern)
    self.data = glob(data_path)

    # validate data
    if len(self.data) is 0:
      raise ValueError('Cannot find data in {}!!!!'.format(data_path))

    self.c_dim = 3
    self.label_dict = {}
    labels_path = '{}/**/'.format(self.sample_data_dir)
    path_list = glob(labels_path, recursive=True)[1:]
    for i, elem in enumerate(path_list):
      dirs = elem.split('/')
      label = dirs[len(dirs)-2]
      self.label_dict[label] = i

    self.experience_buffer=[]
    self.grayscale = (self.c_dim == 1)

    self.build_model(old_model=old_model)

  def upsample(
    self,
    input_,
    output_shape,
    k_h=5,
    k_w=5,
    d_h=2,
    d_w=2,
    stddev=0.02,
    name=None
    ):
      if self.use_resize:
        return resizeconv(
          input_=input_,
          output_shape=output_shape,
          k_h=k_h,
          k_w=k_w,
          d_h=d_h,
          d_w=d_w,
          name=(name or "resconv")
        )

      return deconv2d(
        input_=input_,
        output_shape=output_shape,
        k_h=k_h,
        k_w=k_w,
        d_h=d_h,
        d_w=d_w,
        name= (name or "deconv2d")
      )

  def make_style_net(self, images):
    with tf.device("/gpu:0"):
      network_fn = nets_factory.get_network_fn(
        'inception_resnet_v2',
        num_classes=27,
        is_training=False
      )
      if images.shape[1:3] != (256, 256):
        images = tf.image.resize_images(images, [256, 256])
      logits, _ = network_fn(images)
      logits = tf.stop_gradient(logits)
      return logits

  def set_sess(self, sess):
    ''' set session to sess '''
    self.sess = sess

  def build_model(self, old_model=False):
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')
    else:
      self.y = None

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [None] + image_dims, name='real_images')

    self.z = tf.placeholder(
      tf.float32,
      [None, self.z_dim],
      name='z'
    )

    self.z_sum = histogram_summary("z", self.z)

    if self.wgan and not self.can:
        self.discriminator = discriminators.dcwgan_cond
        self.generator = generators.dcgan_cond
        self.d_update, self.g_update, self.losses, self.sums = WGAN_loss(self)

    if self.wgan and self.can:
        self.discriminator = discriminators.vanilla_wgan
        self.generator = generators.vanilla_wgan
        #TODO: write all this wcan stuff
        self.d_update, self.g_update, self.losses, self.sums = WCAN_loss(self)

    if not self.wgan and self.can:
        self.discriminator = discriminators.vanilla_can
        self.generator = generators.vanilla_can
        self.d_update, self.g_update, self.losses, self.sums = CAN_loss(self)

    elif not self.wgan and not self.can:
        #TODO: write the regular gan stuff
        self.d_update, self.g_update, self.losses, self.sums = GAN_loss(self)

    if self.can or not self.y_dim:
        self.sampler = self.generator(self, self.z, is_sampler=True)
    else:
        self.sampler = self.generator(self, self.z, self.y, is_sampler=True)

    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    if self.style_net_checkpoint:
      all_vars = tf.trainable_variables()
      style_net_vars = [v for v in all_vars if 'InceptionResnetV2' in v.name]
      other_vars = [v for v in all_vars if 'InceptionResnetV2' not in v.name]
      self.saver = tf.train.Saver(var_list=other_vars)
      self.style_net_saver = tf.train.Saver(var_list=style_net_vars)
    else:
      self.saver=tf.train.Saver()

  def train(self, config):
    print(" [*] Training")
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.log_dir = config.log_dir

    self.writer = SummaryWriter(self.log_dir, self.sess.graph)

    # generate random noise
    sample_z = np.random.normal(0, 1, [self.sample_num, self.z_dim]).astype(np.float32)
    sample_z /= np.linalg.norm(sample_z, axis=0)
    sample_files = self.data[0:self.sample_num]
    sample = [
        get_image(
          sample_file,
          input_height=self.input_height,
          input_width=self.input_width,
          resize_height=self.output_height,
          resize_width=self.output_width,
          crop=self.crop,
          grayscale=self.grayscale
        ) for sample_file in sample_files]

    if (self.grayscale):
      sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
      sample_inputs = np.array(sample).astype(np.float32)
    
    sample_labels = self.get_y(sample_files)

    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter, loaded_sample_z = self.load(
      self.checkpoint_dir,
      config,
      style_net_checkpoint_dir=self.style_net_checkpoint
    )

    if could_load:
      counter = checkpoint_counter
      if self.replay:
        replay_files = glob(os.path.join(self.model_dir + '_replay'))
        self.experience_buffer = [
          get_image(sample_file,
            input_height=self.input_height,
            input_width=self.input_width,
            resize_height=self.output_height,
            resize_width=self.output_width,
            crop=self.crop,
            grayscale=self.grayscale
          ) for sample_file in replay_files
        ]
      print(" [*] Load SUCCESS")
      if loaded_sample_z is not None:
        sample_z = loaded_sample_z
    else:
      print(" [!] Load failed...")

    np.save(os.path.join(self.checkpoint_dir, 'sample_z'), sample_z)

    for epoch in xrange(config.epoch):
      print(" [{}] Training epoch".format(epoch))

      shuffle(self.data)
      batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        self.experience_flag = not bool(idx % 2)

        batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
        batch = [
            get_image(batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=self.crop,
                      grayscale=self.grayscale) for batch_file in batch_files]
        if self.grayscale:
          batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
        else:
          batch_images = np.array(batch).astype(np.float32)
        batch_labels = self.get_y(batch_files)

        batch_z = np.random.normal(0, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)
        batch_z /= np.linalg.norm(batch_z, axis=0)

        if self.can:
          
          # update Discriminator
          _, summary_str = self.sess.run([self.d_update, self.sums[0]],
            feed_dict={
              self.inputs: batch_images,
              self.z: batch_z,
              self.y: batch_labels,
            })
          self.writer.add_summary(summary_str,counter)
          
          # update Generator -- don't need labels or inputs
          _, summary_str = self.sess.run([self.g_update, self.sums[1]],
            feed_dict={
              self.z: batch_z,

            })
          self.writer.add_summary(summary_str, counter)
          
          # do we need self.y for these two?
          errD_fake = self.d_loss_fake.eval({
              self.z: batch_z,
              self.y:batch_labels
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
              self.y:batch_labels
          })
          errG = self.g_loss.eval({
              self.z: batch_z
          })

          errD_class_real = self.d_loss_class_real.eval({
              self.inputs: batch_images,
              self.y: batch_labels
          })
          errG_class_fake = self.g_loss_class_fake.eval({
              self.inputs: batch_images,
              self.z: batch_z
          })
          accuracy = self.accuracy.eval({
              self.inputs: batch_images,
              self.y: batch_labels
          })
        else:
          # Update D network
          if self.wgan:
            for i in range(4):
              _, summary_str = self.sess.run([self.d_update, self.d_sum],
                feed_dict={
                  self.inputs: batch_images,
                  self.z: batch_z,
                  self.y: batch_labels,
              })
              self.writer.add_summary(summary_str, counter)
              slopes = self.sess.run(self.slopes,

                feed_dict={
                  self.inputs: batch_images,
                  self.z: batch_z,
                  self.y: batch_labels

              })
          _, summary_str = self.sess.run([self.d_update, self.d_sum],
            feed_dict={
              self.inputs: batch_images,
              self.z: batch_z,
              self.y:batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([self.g_update, self.g_sum],
            feed_dict={
              self.z: batch_z,
              self.y: batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          errD = self.d_loss.eval({
              self.inputs: batch_images,
              self.y:batch_labels,
              self.z:batch_z
          })
          errG = self.g_loss.eval({
              self.z: batch_z,
              self.y: batch_labels
          })

        counter += 1
        if self.can:
          print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
            % (epoch, idx, batch_idxs,
              time.time() - start_time, errD_fake+errD_real+errD_class_real, errG))
          print("Discriminator class acc: %.2f" % (accuracy))
        else:
          if self.wgan:
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
            % (epoch, idx, batch_idxs,
              time.time() - start_time, errD, errG))
          else:
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
            % (epoch, idx, batch_idxs,
              time.time() - start_time, errD, errG))

        if np.mod(counter, 5) == 1 and self.replay:
          samp_images = self.G.eval({
              self.z: batch_z
          })
          if self.experience_flag:
            exp_path = os.path.join('buffer', self.model_dir)
            #max_ = get_max_end(exp_path)
            for i, image in enumerate(samp_images):
              #scipy.misc.imsave(exp_path + '_' + str(max_+i) + '.jpg', np.squeeze(image))
              self.experience_buffer.append(image)
            # todo make into a flag
            exp_buffer_len = 10000
            if len(self.experience_buffer) > exp_buffer_len:
              self.experience_buffer = self.experience_buffer[len(self.experience_buffer) - exp_buffer_len:]


        if np.mod(counter, config.sample_itr) == 1:

          samples = self.sess.run(
            self.sampler,
            feed_dict={
                self.z: sample_z,
                self.inputs: sample_inputs,
                self.y:sample_labels,
            }
          )
          save_images(
            samples,
            image_manifold_size(samples.shape[0]),
            './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx)
          )

        if np.mod(counter, config.save_itr) == 2:
          self.save(config.checkpoint_dir, counter, config)


  def get_y(self, sample_inputs):
    ret = []
    for sample in sample_inputs:
      directories = sample.split('/')
      # assume the label is the file's parent directory
      label = directories[len(directories) - 2]
      ret.append(np.eye(self.y_dim)[np.array(self.label_dict[label])])
    return ret

  @property
  def model_dir(self):
    return "./models"

  def save(self, checkpoint_dir, step, config):
    model_name = "DCGAN.model"
    if not config.use_default_checkpoint:
      checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

    if config.use_s3:
      s3_dir = checkpoint_dir
      aws.upload_path(checkpoint_dir, config.s3_bucket, s3_dir)
      print('uploading log')
      aws.upload_path(self.log_dir, config.s3_bucket, self.log_dir, certain_upload=True)


  def load_specific(self, checkpoint_dir):
    ''' like loading but takes in a directory directly'''
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0

  def load(
    self,
    checkpoint_dir,
    config,
    style_net_checkpoint_dir=None,
    use_last_checkpoint=True
  ):
    print(" [*] Reading checkpoints...")
    if not config.use_default_checkpoint:
      checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if style_net_checkpoint_dir is not None:
      ckpt = tf.train.get_checkpoint_state(style_net_checkpoint_dir)
      if not ckpt:
        raise ValueError('style_net_checkpoint_dir points to wrong directory/model doesn\'t exist')
      ckpt_name = os.path.join(style_net_Fcheckpoint_dir, os.path.basename(ckpt.model_checkpoint_path))
      self.style_net_saver.restore(self.sess, tf.train.latest_checkpoint(style_net_checkpoint_dir))

    # finds the checkpoint
    if config.use_default_checkpoint and use_last_checkpoint:
      path = get_parent_path(get_parent_path( checkpoint_dir))
      #find the high checkpoint path in a path
      files_in_path = sorted(os.listdir(path))

      if len(files_in_path) > 1:
        last_ = files_in_path[-2]
        checkpoint_dir  = os.path.join(path, last_, 'checkpoint')
      else:
        checkpoint_dir = None

    if config.load_dir:
      checkpoint_dir = config.load_dir

    if checkpoint_dir:
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        if os.path.exists(os.path.join(checkpoint_dir, 'sample_z.npy')):
          print(" [*] Success to read sample_z in {}".format(ckpt_name))
          sample_z = np.load(os.path.join(checkpoint_dir, 'sample_z.npy'))
        else:
          print(" [*] Failed to find a sample_z")
          sample_z = None
        return True, counter, sample_z
      elif config.load_dir:
        raise ValueError(" [*] Failed to find the load_dir")

    print(" [*] Failed to find a checkpoint")
    return False, 0, None


