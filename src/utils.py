from __future__ import division
import os
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
from glob import glob

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()


def get_stddev(x, k_h, k_w):
    shape = k_w * k_h * x.get_shape()
    return 1 / math.sqrt(shape[-1])


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_image(
    image_path,
    input_height,
    input_width,
    resize_height=64,
    resize_width=64,
    crop=True,
    grayscale=False
):  
    image = imread(image_path, grayscale)
    try:
        return transform(
            image,
            input_height,
            input_width,
            resize_height,
            resize_width,
            crop
        )
    except ValueError:
        print("Bad image. filepath: ", image_path)


def save_images(images, size, image_path):
    return imsave(
        inverse_transform(images),
        size,
        image_path
    )


def imread(path, grayscale = False):
    try:
        if grayscale:
            return scipy.misc.imread(path, flatten = True).astype(np.float)
        else:
            return scipy.misc.imread(path).astype(np.float)
    except(TypeError):
        print(path)


def test_images(path_glob):
    for path in path_glob:
        imread(path)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('merge() `images` arg must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def center_crop(
    x,
    crop_h,
    crop_w,
    resize_h=64,
    resize_w=64
):
    # if crop width is not specified, use a 1:1 aspect ratio
    if crop_w is None:
        crop_w = crop_h

    height, width = x.shape[:2]
    j = int(round((height - crop_h)/2.))
    i = int(round((width - crop_w)/2.))

    resized = scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

    return resized


def transform(
    image,
    input_height,
    input_width,
    resize_height=64,
    resize_width=64,
    crop=True
):

    cropped_image = (
        center_crop(image, input_height, input_width, resize_height, resize_width)
        if crop else
        scipy.misc.imresize(image, [resize_height, resize_width])
    )
    
    return np.array(cropped_image)/127.5 - 1.


def inverse_transform(images):
  return (images + 1.)/2.


def visualize(sess, dcgan, config):
    image_frame_dim = int(math.ceil(config.batch_size**.5))
    z_sample = np.random.normal(0, 1, size=(config.batch_size, dcgan.z_dim))
    z_sample /= np.linalg.norm(z_sample, axis=0)
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    path = './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime())
    save_images(samples, [image_frame_dim, image_frame_dim], path)


def get_max_end(path_dir, num_len=3, fname_pattern='*.jpg'):
    max_ = 0
    for f in glob(path_dir + fname_pattern):
        curr = int(f[-num_len-4:-4])
        if curr > max_:
            max_ = curr
    return max_


def image_manifold_size(num_images):
    print(num_images)
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


def get_parent_path(path):
    return os.path.normpath(os.path.join(path, os.pardir))
