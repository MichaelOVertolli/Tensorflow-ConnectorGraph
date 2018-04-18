###############################################################################
#Copyright (C) 2017  Michael O. Vertolli michaelvertolli@gmail.com
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see http://www.gnu.org/licenses/
###############################################################################


import json
import os
from PIL import Image
from glob import glob
import tensorflow as tf
from models.friqa import toyiq
from models.errors import ConfigError
import numpy as np
import re


Dataset = tf.data.Dataset
Iterator = tf.data.Iterator

def get_loader(root, batch_size, scale_size, data_format, split=None, is_grayscale=False, seed=None):
    dataset_name = os.path.basename(root)
    if any([x in dataset_name for x in  ['CelebA', 'lsun', 'msceleb', 'oxfordflower', 'imgnet', 'grass', 'brick', 'pflowers', 'celeb', 'mnist']]) and split:
        root = os.path.join(root, 'splits', split)

    for ext in ["jpg", "JPEG", "png"]:
        paths = glob("{}/*.{}".format(root, ext))

        if ext == "jpg" or ext == "JPEG":
            tf_decode = tf.image.decode_jpeg
        elif ext == "png":
            tf_decode = tf.image.decode_png
        if len(paths) != 0:
            break

    with Image.open(paths[0]) as img:
        w, h = img.size
        if is_grayscale:
            shape = [h, w, 1]
        else:
            shape = [h, w, 3]
        

    filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf_decode(data, channels=3)

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size

    queue = tf.train.shuffle_batch(
        [image], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    if dataset_name in ['CelebA']:
        queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    elif any([x in dataset_name for x in  ['grass', 'brick', 'pflowers', 'celeb', 'mnist']]):
        pass
    else:
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])

    if data_format == 'NCHW':
        queue = tf.transpose(queue, [0, 3, 1, 2])
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))

    #for yiq image
    #return tf.stack(toyiq(tf.to_float(queue)/255.), axis=1)*255.
    return tf.to_float(queue)


def shape_from_name(fname):
    m = re.search('(?<=_h)\d+', fname)
    if m is not None:
        h = int(m.group(0))
    else:
        raise ConfigError('Invalid height in filename: {}.'.format(fname))
    m = re.search('(?<=_w)\d+', fname)
    if m is not None:
        w = int(m.group(0))
    else:
        raise ConfigError('Invalid width in filename: {}.'.format(fname))
    m = re.search('(?<=_c)\d+', fname)
    if m is not None:
        c = int(m.group(0))
    else:
        raise ConfigError('Invalid channel in filename: {}.'.format(fname))
    return h, w, c


def setup_sharddata(data_dir, fetch_size, batch_size=16, repeat=0,
                    greyscale=False, norm=False, shuffle=False,
                    bool_masks=None, resize=None, data_format='NCHW'):
    paths = sorted(list(glob('{}/*.tfrecords'.format(data_dir))))
    shard_count = len(paths)
    data = Dataset.from_tensor_slices(tf.constant(paths))

    if bool_masks is not None:
        bools = tf.constant(bool_masks)
        data = Dataset.zip((data, Dataset.range(len(paths))))

    with open(os.path.join(data_dir, 'img_shape.json'), 'r') as f:
        h, w, c = json.loads(f.read())

    if greyscale:
        c = 1
    
    def parse_tf_record(serialized):
        features = tf.parse_single_example(
            serialized,
            features={
                'index': tf.FixedLenFeature((), tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
            })
        i = features['index']
        img = tf.image.decode_jpeg(features['image_raw'], channels=c)
        img.set_shape([h, w, c])
        img = tf.to_float(img)
        return i, img

    if shuffle:
        data = data.shuffle(shard_count)
        if bool_masks is not None:
            data = data.interleave(lambda f, i: Dataset.zip((tf.data.TFRecordDataset(f).map(parse_tf_record),
                                                             Dataset.from_tensor_slices(bools[i]))),
                                   cycle_length=shard_count)
            data = data.filter(lambda d, b: b)
            data = data.map(lambda d, b: d)
        else:
            data = data.interleave(lambda f: tf.data.TFRecordDataset(f).map(parse_tf_record),
                                   cycle_length=shard_count)
        data = data.shuffle(fetch_size)
    else:
        if bool_masks is not None:
            data = data.flat_map(lambda f, i: Dataset.zip((tf.data.TFRecordDataset(f).map(parse_tf_record),
                                                           Dataset.from_tensor_slices(bools[i]))))
            data = data.filter(lambda d, b: b)
            data = data.map(lambda d, b: d)
        else:
            data = data.flat_map(lambda f: tf.data.TFRecordDataset(f).map(parse_tf_record))


    if repeat != 0:
        data = data.repeat(repeat)

    data = data.map(lambda index, imgs: imgs)
    
    data = data.batch(batch_size)

    if resize is not None:
        data = data.map(lambda imgs: tf.image.resize_bicubic(imgs, resize))

    if data_format == 'NCHW':
        data = data.map(lambda imgs: tf.transpose(imgs, [0, 3, 1, 2]))

    if norm:
        data = data.map(lambda imgs: (imgs/127.5) - 1.0)

    data = data.prefetch(fetch_size/batch_size)

    itr = Iterator.from_structure(data.output_types, data.output_shapes)
    loader = itr.get_next()
    init = itr.make_initializer(data)

    return data, loader, init


def setup_nameddata(fname, fetch_size, batch_size=16,
                    greyscale=False, norm=False,resize=None, data_format='NCHW'):
    data = tf.data.TFRecordDataset(fname)
    data_dir, _ = os.path.split(fname)

    with open(os.path.join(data_dir, 'img_shape.json'), 'r') as f:
        h, w, c = json.loads(f.read())

    if greyscale:
        c = 1
    
    def parse_tf_record_name(serialized):
        features = tf.parse_single_example(
            serialized,
            features={
                'image_name': tf.FixedLenFeature([], tf.string),
            })
        name = features['image_name']
        return name

    def parse_tf_record_img(serialized):
        features = tf.parse_single_example(
            serialized,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
            })
        img = tf.image.decode_jpeg(features['image_raw'], channels=c)
        img.set_shape([h, w, c])
        img = tf.to_float(img)
        return img

    names = data.map(parse_tf_record_name)
    imgs = data.map(parse_tf_record_img)
    
    names = names.batch(batch_size)
    imgs = imgs.batch(batch_size)

    if resize is not None:
        imgs = imgs.map(lambda imgs: tf.image.resize_bicubic(imgs, resize))

    if data_format == 'NCHW':
        imgs = imgs.map(lambda imgs: tf.transpose(imgs, [0, 3, 1, 2]))

    if norm:
        imgs = imgs.map(lambda imgs: (imgs/127.5) - 1.0)

    data = Dataset.zip((names, imgs))
    data = data.prefetch(fetch_size/batch_size)

    itr = Iterator.from_structure(data.output_types, data.output_shapes)
    name_loader, img_loader = itr.get_next()
    init = itr.make_initializer(data)

    return data, name_loader, img_loader, init


def setup_rdataset(fname, fetch_size, batch_size=16, repeat=0,
                   greyscale=False, norm=False, shuffle=False,
                   bool_mask=None, resize=None, data_format='NCHW'):
    data = tf.data.TFRecordDataset(fname)

    if repeat != 0:
        data = data.repeat(repeat)

    h, w, c = shape_from_name(fname)

    if greyscale:
        c = 1

    def parse_tf_record(serialized):
        features = tf.parse_single_example(
            serialized,
            features={
                'index': tf.FixedLenFeature((), tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
            })
        i = features['index']
        img = tf.image.decode_jpeg(features['image_raw'], channels=c)
        img.set_shape([h, w, c])
        img = tf.to_float(img)
        return i, img
    
    data = data.map(parse_tf_record)

    if bool_mask is not None:
        bools = Dataset.from_tensor_slices(tf.constant(bool_mask))
        if repeat != 0:
            bools.repeat(repeat)
        data = Dataset.zip((data, bools))
        data = data.filter(lambda data, bool_: bool_)
        data = data.map(lambda data, bools: data)

    data = data.map(lambda index, imgs: imgs)

    if shuffle:
        data = data.shuffle(fetch_size)
    
    data = data.batch(batch_size)

    if resize is not None:
        data = data.map(lambda imgs: tf.image.resize_bicubic(imgs, resize))

    if data_format == 'NCHW':
        data = data.map(lambda imgs: tf.transpose(imgs, [0, 3, 1, 2]))

    if norm:
        data = data.map(lambda imgs: (imgs/127.5) - 1.0)

    data = data.prefetch(fetch_size/batch_size)

    itr = Iterator.from_structure(data.output_types, data.output_shapes)
    loader = itr.get_next()
    init = itr.make_initializer(data)

    return data, loader, init


def setup_dataset(root, batch_size=16, shuffle=False, repeat=0, greyscale=False,
                 file_format='jpg', data_format='NCHW'):
    paths = sorted(list(glob('{}/*.{}'.format(root, file_format))))
    with Image.open(paths[0]) as img:
        w, h = img.size
    if greyscale:
        c = 1
    else:
        c = 3
    shape = [h, w, c]
    data = Dataset.from_tensor_slices(tf.constant(paths))

    def img_parser(img_path):
        img_data = tf.read_file(img_path)
        img = tf.image.decode_jpeg(img_data, channels=c)
        img.set_shape(shape)
        return img
    
    data = data.map(img_parser)
    data = data.batch(batch_size)
    
    if repeat != 0:
        data = data.repeat(repeat)

    if shuffle:
        data = data.shuffle()

    def img_mod(batch_imgs):
        if data_format == 'NCHW':
            batch_imgs = tf.transpose(batch_imgs, [0, 3, 1, 2])
        batch_imgs = tf.to_float(batch_imgs)
        # batch_imgs = (batch_imgs/127.5) - 1.0
        return batch_imgs

    data = data.map(img_mod)

    itr = Iterator.from_structure(data.output_types, data.output_shapes)
    nxt = itr.get_next()
    init = itr.make_initializer(data)

    return paths, nxt, init


def setup_queue(root, batch_size=16, shuffle=False, num_epochs=1, greyscale=False,
                 file_format='jpg', data_format='NCHW'):
    paths = sorted(list(glob('{}/*.{}'.format(root, file_format))))
    fqueue = tf.train.string_input_producer(paths, num_epochs, shuffle=False)
    reader = tf.WholeFileReader()
    fname, data = reader.read(fqueue)

    with Image.open(paths[0]) as img:
        w, h = img.size

    if num_epochs != 0:
        allow_smaller_final_batch = True
    else:
        allow_smaller_final_batch = False
    if greyscale:
        image = tf.image.decode_jpeg(data, channels=1)
        shape = [h, w, 1]
    else:
        image = tf.image.decode_jpeg(data, channels=3)
        shape = [h, w, 3]
    image.set_shape(shape)
    if shuffle:
        min_after_dequeue = 5000
        capacity = min_after_dequeue + 3 * batch_size
        queue = tf.train.shuffle_batch(
            [image], batch_size=batch_size,
            num_threads=4, capacity=capacity,
            allow_smaller_final_batch=allow_smaller_final_batch,
            min_after_dequeue=min_after_dequeue, name='synthetic_inputs')
    else:
        capacity = 100 * batch_size
        queue = tf.train.batch(
            [image], batch_size=batch_size,
            num_threads=1, capacity=capacity,
            allow_smaller_final_batch=allow_smaller_final_batch,
            name='synthetic_inputs')

    queue = tf.transpose(queue, [0, 3, 1, 2])
    queue = tf.to_float(queue)
    return paths, queue
    

def setup_coord(sess, queue_runner):
    coord = tf.train.Coordinator()
    enqueue_threads = queue_runner.create_threads(sess, coord=coord, start=True)
    return coord

