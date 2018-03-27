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

from glob import glob
import json
from nltk.corpus import wordnet as wn
import numpy as np
import os
from PIL import Image
from random import randint, shuffle
import tensorflow as tf
import xml.etree.ElementTree as ET

def get_hyponyms(syn):
    hypo = set()
    for h in syn.hyponyms():
        hypo |= set(get_hyponyms(h))
    return hypo | set(syn.hyponyms())


def get_syn_imgs(syn, dir_):
    dirs = os.listdir(dir_)
    hypos = get_hyponyms(syn)
    wnids = dict([('n{:08}'.format(s.offset()), True) for s in hypos])
    out = []
    for d in dirs:
        try:
            temp = wnids[d.split('_')[0]]
        except KeyError:
            continue
        else:
            out.append(d)
    return out


def get_bbox(file_, from_dir):
    wnid = file_.split('_')[0]
    xml = file_[:-4]+'xml'
    path = os.path.join(from_dir, wnid, xml)
    try:
        root = ET.parse(path).getroot()
    except IOError:
        out = None
    else:
        box = root[5][4]
        min_ = [int(box[0].text), int(box[1].text)]
        max_ = [int(box[2].text), int(box[3].text)]
        size = root[3]
        img_size = [int(size[0].text), int(size[1].text)]
        out = min_, max_, img_size
    return out


def square_bbox(min_, max_, size):
    bsize = [max_[0]-min_[0], max_[1]-min_[1]]
    if bsize[0] < bsize[1]:
        i = 0
        diff = bsize[1] - bsize[0]
    else:
        i = 1
        diff = bsize[0] - bsize[1]
    if diff+bsize[i] > size[i]:
        out = None
    else:
        if diff%2 == 0:
            ltdiff = diff/2
            rbdiff = ltdiff
        else:
            ltdiff = diff/2
            rbdiff = ltdiff+1
        mn = [x for x in min_]
        mx = [x for x in max_]
        mn[i] -= ltdiff
        mx[i] += rbdiff
        if mn[i] < 0:
            mx[i] -= mn[i]
            mn[i] = 0
        if mx[i] > size[i]:
            mn[i] -= mx[i] - size[i]
            mx[i] = size[i]
        out =  [mn[0], mn[1], mx[0], mx[1]]
    return out


def save_bbox_crop_imgs(syn, imgs_dir, annt_dir, out_dir):
    files = get_syn_imgs(syn, imgs_dir)
    for f in files:
        bbox = get_bbox(f, annt_dir)
        if bbox is None:
            continue
        min_, max_, size = bbox
        crop_pos = square_bbox(min_, max_, size)
        if crop_pos is None:
            continue
        im = Image.open(os.path.join(imgs_dir, f))
        crop = im.crop(crop_pos)
        resized = crop.resize((256, 256), Image.BICUBIC)
        resized.save(os.path.join(out_dir, f))
        im.close()


def symlink_imgs(files, from_dir, to_dir):
    for f in files:
        os.symlink(os.path.join(from_dir, f), os.path.join(to_dir, f))


def imgnetmain():
    item = wn.synsets('animal')[0]
    imgs_dir = './train/'
    annt_dir = '/home/olias/data/imgnet/annotations/'
    out_dir = '/home/olias/data/imgnet_animal/splits/train/'
    save_bbox_crop_imgs(item, imgs_dir, annt_dir, out_dir)


def crop_resize(imgs_dir, out_dir):
    files = os.listdir(imgs_dir)
    for f in files:
        im = Image.open(os.path.join(imgs_dir, f))
        h, w = im.size
        if h > w:
            crop = im.crop((0, 0, w, w))
        else:
            crop = im.crop((0, 0, h, h))
        resized = crop.resize((128, 128), Image.BICUBIC)
        resized.save(os.path.join(out_dir, f))
        im.close()


def mscelebmain():
    crop_resize('/home/olias/data/msceleb/MsCeleb', '/home/olias/data/msceleb/splits/train')


def dataset_from_img(img_file, dataset_folder, dataset_size, out_img_size):
    im = Image.open(img_file)
    w, h = im.size
    w -= (out_img_size + 1)
    h -= (out_img_size + 1)
    for i in range(dataset_size):
        w_ = randint(0, w)
        h_ = randint(0, h)
        crop = im.crop((w_, h_, w_+out_img_size, h_+out_img_size))
        crop.save(os.path.join(dataset_folder, '{:07}.jpg'.format(i)))
    im.close()


def dataset_resize(imgs_dir, new_dir, new_size):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    files = os.listdir(imgs_dir)
    for f in files:
        im = Image.open(os.path.join(imgs_dir, f))

        o_im = im.resize([new_size, new_size], Image.NEAREST)
        o_im.save(os.path.join(new_dir, f))

        im.close()


def mnist_to_imgs(new_dir, base_size=32):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    images = mnist.train.images
    for i in range(images.shape[0]):
        arr = np.reshape(images[i, :], [28, 28])
        im = Image.fromarray(arr*255, 'I').convert('RGB')
        o_im = im.resize([base_size, base_size], Image.NEAREST)
        o_im.save(os.path.join(new_dir, '{:05}.jpg'.format(i)))


def cifar10_to_imgs(new_dir):
    files = glob('/home/olias/data/cifar-10-batches-bin/data_batch*.bin')
    
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    label_bytes = 1
    h, w, c = 32, 32, 3
    image_bytes = h * w * c

    convert = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    counts = dict([(s, 0) for s in convert])
    convert = dict([(i, s) for i, s in enumerate(convert)])

    for f in files:
        with open(f, 'rb') as stream:
            label_i = stream.read(label_bytes)
            while label_i != '':
                label_i = np.frombuffer(label_i, dtype=np.uint8)[0]
                label = convert[label_i]
                index = counts[label]
                counts[label] += 1
                img = np.frombuffer(stream.read(image_bytes), dtype=np.int8)
                img = np.transpose(np.reshape(img, [c, w, h]), [1, 2, 0]).astype(np.uint8)
                img = Image.fromarray(img)
                img.save(os.path.join(new_dir, '{}_{}.jpg'.format(label, index)))
                label_i = stream.read(label_bytes)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def to_tfrecord(fname, new_dir, img_dir, shape, shards=25): # assumes shape is [h, w, c]
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    with open(os.path.join(new_dir, 'img_shape.json'), 'w') as f:
        f.write(json.dumps(shape))
    imgs = glob(os.path.join(img_dir, '*.jpg'))
    shuffle(imgs)
    nimgs_pershard = len(imgs) / shards 
    imgs = imgs[:nimgs_pershard*shards] # cuts off remainder < num shards
    print '{} images kept after filtering.'.format(len(imgs))

    shardstart_i = 0
    shard_i = 0
    writer = tf.python_io.TFRecordWriter(os.path.join(new_dir, fname+'_{:02}.tfrecords'.format(shard_i)))
    for i, img in enumerate(imgs):
        if (i - shardstart_i) == nimgs_pershard:
            writer.close()
            shardstart_i = i
            shard_i += 1
            writer = tf.python_io.TFRecordWriter(os.path.join(new_dir, fname+'_{:02}.tfrecords'.format(shard_i)))
        
        with open(img, 'rb') as f:
            im_raw = f.read()

        example = tf.train.Example(features=tf.train.Features(feature={
            'index': _int64_feature(i),
            'image_raw': _bytes_feature(im_raw)}))

        writer.write(example.SerializeToString())
        
    writer.close()
