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

from evaluations import flatten2color
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


def crop_resize2(imgs_dir, out_dir, crop_box, resize):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    files = os.listdir(imgs_dir)
    for f in files:
        im = Image.open(os.path.join(imgs_dir, f))
        h, w = im.size
        if h > w:
            crop = im.crop(crop_box)
        else:
            crop = im.crop(crop_box)
        if resize is not None:
            resized = crop.resize(resize, Image.BICUBIC)
        else:
            resized = crop
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


def celeb_to_imgs(new_dir, imgs_dir='/home/olias/data/img_align_celeba/', train=True, all_dir=None):
    if all_dir is None:
        dirs = [new_dir]
    else:
        dirs = [new_dir, all_dir]
    for dr in dirs:
        if not os.path.exists(dr):
            os.makedirs(dr)
    TRAIN_STOP = 162770
    NUM_EXAMPLES = 202599
    CROP_BOX = [25, 50, 128+25, 50+128]
    if train:
        start, stop = 0, TRAIN_STOP
    else:
        start, stop = TRAIN_STOP, NUM_EXAMPLES # collapsedd validation to test
    
    files = ['{:06}.jpg'.format(i+1) for i in range(start, stop)]
    
    for f in files:
        im = Image.open(os.path.join(imgs_dir, f))
        crop = im.crop(CROP_BOX)
        for dr in dirs:
            crop.save(os.path.join(dr, f))
        im.close()


def mnist_to_imgs(new_dir, base_size=32, train=True, all_dir=None):
    if all_dir is None:
        dirs = [new_dir]
    else:
        dirs = [new_dir, all_dir]
    for dr in dirs:
        if not os.path.exists(dr):
            os.makedirs(dr)
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    if train:
        images = mnist.train.images
        labels = mnist.train.labels
        base_index = 0
    else:
        images = mnist.test.images
        labels = mnist.test.labels
        base_index = 55000
    labels = np.asarray(labels, dtype=np.int32)
    for i in range(images.shape[0]):
        arr = np.reshape(images[i, :], [28, 28])
        im = Image.fromarray(arr*255, 'I').convert('RGB')
        o_im = im.resize([base_size, base_size], Image.NEAREST)
        label = labels[i]
        for dr in dirs:
            o_im.save(os.path.join(dr, '{}_{:05}.jpg'.format(label, base_index+i)))


def cifar10_to_imgs(new_dir, train=True, all_dir=None):
    if all_dir is None:
        dirs = [new_dir]
    else:
        dirs = [new_dir, all_dir]
    for dr in dirs:
        if not os.path.exists(dr):
            os.makedirs(dr)
    if train:
        files = glob('/home/olias/data/cifar-10-batches-bin/data_batch*.bin')
        count = 0
    else:
        files = ['/home/olias/data/cifar-10-batches-bin/test_batch.bin']
        count = 50000

    label_bytes = 1
    h, w, c = 32, 32, 3
    image_bytes = h * w * c

    # convert = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # convert = dict([(i, s) for i, s in enumerate(convert)])

    for f in files:
        with open(f, 'rb') as stream:
            label_i = stream.read(label_bytes)
            while label_i != '':
                label_i = np.frombuffer(label_i, dtype=np.uint8)[0]
                # label = convert[label_i]
                img = np.frombuffer(stream.read(image_bytes), dtype=np.int8)
                img = np.transpose(np.reshape(img, [c, w, h]), [1, 2, 0]).astype(np.uint8)
                img = Image.fromarray(img)
                for dr in dirs:
                    img.save(os.path.join(dr, '{}_{:05}.jpg'.format(label_i, count)))
                count += 1
                label_i = stream.read(label_bytes)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def to_tfrecord_shard(fname, new_dir, img_dir, shape, shards=25): # assumes shape is [h, w, c]
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


def to_tfrecord(fname, new_dir, img_dir, shape, shuffle=False): # assumes shape is [h, w, c]
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    with open(os.path.join(new_dir, 'img_shape.json'), 'w') as f:
        f.write(json.dumps(shape))
    imgs = glob(os.path.join(img_dir, '*.jpg'))
    if shuffle:
        shuffle(imgs)

    writer = tf.python_io.TFRecordWriter(os.path.join(new_dir, fname+'.tfrecords'))
    for i, img in enumerate(imgs):
        with open(img, 'rb') as f:
            im_raw = f.read()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(im_raw),
            'image_name': _bytes_feature(os.path.split(img)[1])}))

        writer.write(example.SerializeToString())
        
    writer.close()


#Param search

def strip_extra_headers(ar):
    new = []
    for i in range(ar.shape[0]):
        if not np.isnan(ar[i][0]):
            new.append(ar[i])
    return np.stack(new)


def to_distance(ar, start_index, end_index):
    new = np.copy(ar)
    temp = new[:, start_index:end_index]
    new[:, start_index:end_index] = np.ones(temp.shape) - temp
    return new


def nan_to_val(ar, nan_val):
    new = np.copy(ar)
    return new


def prep_param_csv(fname, start_index, end_index=14, nan_val=-.1, log=True):
    path, tail = os.path.split(fname)
    with open(fname, 'r') as f:
        header = f.readline()
        ar = np.genfromtxt(f, np.float32, delimiter=',')
    ar = strip_extra_headers(ar)
    # ar = to_distance(ar, start_index, end_index)
    if log:
        ar = -np.log10(ar+1e-10)
        if start_index == 3:
            ar[ar < 0.] = 10**(-ar[ar < 0.])
            ar[ar > 9.] = 0.
    ar[np.isnan(ar)] = nan_val
    np.savetxt(os.path.join(path, 'prepped_'+tail), ar, fmt='%.4f', delimiter=',', header=header[:-1], comments='')


def get_all_colors(log_dir, size=[32, 32], quantize=4, all_colors=set([])):
    files = os.listdir(log_dir)
    for f in files:
        im = Image.open(os.path.join(log_dir, f))
        if size is not None:
            im = im.resize(size)
        im = np.array(im)
        if quantize is not None:
            im = (im / quantize) * quantize
        all_colors |= set([tuple(c) for c in flatten2color(im).tolist()])
    return all_colors
