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

from collections import OrderedDict
import json
import matplotlib.pyplot as plt
from models.friqa import prep_and_call_qs_sim
from models.model_utils import *
import numpy as np
import pickle
from PIL import Image
from pyemd import emd
from random import shuffle
import re
from scipy.spatial import distance_matrix
from sklearn.manifold import TSNE
from skimage.color import deltaE_ciede2000 as ciede2000
from skimage.color import rgb2lab
from skimage import img_as_ubyte
import tensorflow as tf
import glob
geval = tf.contrib.gan.eval



def comparisons(base, modified, greyscale=False):
    outputs = []
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope('comparisons') as vs:
            b = tf.placeholder(tf.float32, [None, None, None, None],
                               name='base')
            m = tf.placeholder(tf.float32, [None, None, None, None],
                               name='modified')
            l1 = tf.reduce_mean(tf.abs(m - b), axis=[1, 2, 3])
            gms, chrom = prep_and_call_qs_sim(b, m)

    if greyscale:
        out = l1
    else:
        out = [l1, gms, chrom]

    with tf.Session(graph=graph) as sess:
        sess.run(tf.variables_initializer(tf.contrib.framework.get_variables(vs)))

        for b_, m_ in zip(base, modified):
            o = sess.run(out, {b: b_, m: m_})
            outputs.append(np.array(o).T)

    outputs = np.concatenate(outputs)
    mean = np.mean(outputs, axis=0)
    std = np.std(outputs, axis=0)

    return [mean, std]


def get_inception_graph(path='/home/olias/data/inception_model/inceptionv1_for_inception_score.pb'):
    return geval.get_graph_def_from_disk(path)


INCEPTION_INPUT = 'Mul:0'
INCEPTION_OUTPUT = 'logits:0'
INCEPTION_FINAL_POOL = 'pool_3:0'


def inception_score(images, data_format='NCHW'):
    inter = []
    outputs = []
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope('comparisons') as vs:
            inpt = tf.placeholder(tf.float32, [None, None, None, None],
                                  name='inpt')
            imgs = to_nhwc(inpt, data_format)
            imgs = tf.image.resize_images(imgs, [299, 299])
            logits = geval.run_image_classifier(imgs, get_inception_graph(),
                                                INCEPTION_INPUT, INCEPTION_OUTPUT)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.variables_initializer(tf.contrib.framework.get_variables(vs)))

        for ims in images:
            lgt = sess.run(logits, {inpt: ims})
            inter.append(lgt)

    all_logits = np.concatenate(inter)

    

    with tf.Session(graph=tf.Graph()) as sess:
        score = geval.classifier_score_from_logits(tf.constant(all_logits))
        score = sess.run(score)

    return score


def frechet_score(rimages, gimages, data_format='NCHW'): # TODO: test if this works
    ginter = []
    rinter = []
    outputs = []
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope('comparisons') as vs:
            inpt = tf.placeholder(tf.float32, [None, None, None, None],
                                  name='inpt')
            imgs = to_nhwc(inpt, data_format)
            imgs = tf.image.resize_images(imgs, [299, 299])
            activations = geval.run_image_classifier(imgs, get_inception_graph(),
                                                INCEPTION_INPUT, INCEPTION_FINAL_POOL)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.variables_initializer(tf.contrib.framework.get_variables(vs)))

        for ims in gimages:
            actvs = sess.run(activations, {inpt: ims})
            ginter.append(actvs)

        for ims in rimages:
            actvs = sess.run(activations, {inpt: ims})
            rinter.append(actvs)

    all_gactivs = np.concatenate(ginter)
    all_ractivs = np.concatenate(rinter)

    with tf.Session(graph=tf.Graph()) as sess:
        score = geval.frechet_classifier_distance_from_activations(tf.constant(all_ractivs), tf.constant(all_gactivs))
        score = sess.run(score)

    return score


def extract_data(log_dir, base='res_train/loss/', types=['d', 'g'], values=['{}_l1', '{}_gms', '{}_chrom']):
    f = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*.movito'))[0]
    itr = tf.train.summary_iterator(f)
    tags = []
    for t in types:
        for v in values:
            tags.append(os.path.join(base, v.format(t)))
    tag_cnt = len(tags)
    dtags = dict([(k, i) for i, k in enumerate(tags)])
    outputs = []
    append = False
    for e in itr:
        o = np.zeros(tag_cnt)
        for v in e.summary.value:
            try:
                i = dtags[v.tag]
            except KeyError:
                continue
            else:
                o[i] = v.simple_value
                append = True
        if append:
            outputs.append(o)
            append = False
    outputs = np.stack(outputs)
    min_ = np.min(outputs, axis=0)
    mean = np.mean(outputs, axis=0)
    std = np.std(outputs, axis=0)
    return tags, min_, mean, std


def write_data(log_dir, model_log_dir, param_set=[('g_lr', 8e-5), ('d_lr', 8e-5)],
               save_file='param_search_results.csv', first_write=False):
    tags, min_, mean, std = extract_data(model_log_dir)
    alltags = []
    values = []
    for t, v in param_set:
        alltags.append(t)
        values.append(v)
    if first_write:
        for type_ in ['_min', '_mean', '_std']:
            alltags.extend([t.split('/')[-1]+type_ for t in tags])
        with open(os.path.join(log_dir, save_file), 'a+') as f:
            f.write(','.join(alltags)+'\n')
    values.extend(min_)
    values.extend(mean)
    values.extend(std)
    with open(os.path.join(log_dir, save_file), 'a+') as f:
        f.write(','.join([str(v) for v in values])+'\n')


def get_last_img(model_log_dir, img_type):
    imgs = glob.glob(os.path.join(model_log_dir, '*_{}.png'.format(img_type)))
    i_vals = [(int(os.path.split(img)[1].split('_')[0]), img) for img in imgs if 'interp' not in img]
    i_vals = sorted(i_vals)
    return i_vals[-1][1]


def norm_log10(v, g=0.5):
    nv = ((g-0.5)-np.log10(v)-3.0)/5.0
    return nv


def norm_log10_inv(nv, g=0.5):
    v = 10.0**(-((nv*5.0)+3.0-(g-0.5)))
    return v


def get_img_data(log_dir, block, param_keys=['glr', 'dlr'], program='train_program.txt', img_type='G', gamma_filter=None):
    data = {}
    walk = os.walk(log_dir)
    basepath = walk.next()[0]
    for path, dirs, files in walk:
        gamma = float(re.search('(?<=g)\d\.\d', path).group(0))
        if gamma_filter is not None and gamma not in gamma_filter:
            continue
        key = re.search('\d{4}_\d{6}', path).group(0)
        try:
            datum = data[key]
        except KeyError:
            data[key] = {'path': path,
                         'params': None,
                         'img': None}
            datum = data[key]
        if program in files:
            with open(os.path.join(path, program), 'r') as f:
                param_set = json.load(f)
            for params in param_set:
                if params['dir'] == block:
                    datum['params'] = [norm_log10(params[pkey]) for pkey in param_keys]
                    break
            datum['params'].append(float(gamma))
        elif '1999_G.png' in files:
            datum['img'] = get_last_img(path, img_type)
    return data, basepath





def grid_spacing(intstart=1, intend=10, expstart=3, expend=9, hard_front=False):
    intdiff = intend-intstart
    expdiff = expend-expstart
    ar = np.flip(np.array([i for i in range(intstart, intend)], np.float64), 0)
    grid = np.tile(ar, expdiff).reshape([expdiff, intdiff])
    for i in range(expdiff):
        grid[i, :] = grid[i, :]*(10**(-(expstart+i)))
    if hard_front:
        grid[0, :] = intstart*(10**(-expstart))
    return grid


def draw_grid(img, grid, pxtrim, ygrid=True, xgrid=True, dp=3):
    h, w, c = img.shape
    h = h - 2*pxtrim
    w = w - 2*pxtrim
    values = norm_log10(grid)
    cnt = values.shape[0]
    if ygrid:
        yv = values*h + pxtrim
        for i in range(cnt):
            index = int(yv[i])
            img[index-dp:index+dp, pxtrim-(dp+0):h+(1*dp)+pxtrim, :] = 0
    if xgrid:
        xv = values*w + pxtrim
        for i in range(cnt):
            index = int(xv[i])
            img[pxtrim-(dp+0):h+(1*dp)+pxtrim, index-dp:index+dp, :] = 0
    return img

def index_imgs(new_size, pxtrim, data):
    index = {}
    h, w, c = new_size
    h = h - 2*pxtrim
    w = w - 2*pxtrim
    for k in data:
        img = data[k]['img']
        if img is None:
            continue
        img = np.array(Image.open(img))#[2:34, 2:34, :]
        glr, dlr, gamma = data[k]['params']
        nh = int(h*(1-glr)) + pxtrim
        nw = int(w*dlr) + pxtrim
        try:
            imgs = index[(nh, nw)]
        except KeyError:
            index[(nh, nw)] = []
            imgs = index[(nh, nw)]
        imgs.append((gamma, img))
    return index


def chop_img(img, size):
    sz = size+2
    imgs = []
    h, w, c = img.shape
    for w_ in range(2, w-1, sz):
        for h_ in range(2, h-1, sz):
            imgs.append(img[h_:h_+size, w_:w_+size, :])
    return imgs


def join_imgs(imgs, side_len, size):
    all_imgs = []
    for im in imgs:
        all_imgs.extend(chop_img(im, size))
    shuffle(all_imgs)
    img_set = all_imgs[:side_len**2]
    indices = zip(range(0, side_len*side_len, side_len), range(side_len, side_len*(side_len+1), side_len))
    new_img = []
    for j,k in indices:
        new_img.append(np.concatenate(all_imgs[j:k], axis=1))
    new_img = np.concatenate(new_img, axis=0)
    return new_img


def plot_imgs(new_size, index, side_len, size, new_img=None):
    if new_img is None:
        img = np.ones(new_size)*255
    else:
        img = new_img
    szh, szw, c = join_imgs([im for gamma, im in index.values()[0]], side_len, size).shape
    szh = szh/2
    szw = szw/2
    for h, w in sorted(index.keys()):
        imgs = [im for gamma, im in index[(h, w)]]
        new_img = join_imgs(imgs, side_len, size)
        img[(h-szh):h+szh, (w-szw):w+szw, :] = new_img
    return img


def img_data_to_lst(data):
    lst = [[], [], [], []]
    for key in data:
        params = data[key]['params']
        try:
            img = np.array(Image.open(data[key]['img']))
        except AttributeError:
            print key, data[key]
            continue
        pair = (params, img)
        glr = params[0]
        if glr > 9e-5:
            i = 0
        elif glr > 9e-6:
            i = 1
        elif glr > 9e-7:
            i = 2
        elif glr > 9e-8:
            i = 3
        else:
            raise ValueError('Range does not match input params.')
        lst[i].append(pair)
    for i in range(len(lst)):
        lst[i] = sorted(lst[i], key=lambda x: x[0][1]) # within row sort by dlr
    return lst


def save_images(fname, pairs): # from https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    fig = plt.figure()
    mx = max([len(row) for row in pairs])
    for i, row in enumerate(pairs):
        for j, (params, img) in enumerate(row):
            pos = i * mx + j
            a = fig.add_subplot(mx+1, 4, pos + 1)
            plt.imshow(img)
            a.set_title('glr {}, dlr {}, gamma {}'.format(*params))
    # fig.set_size_inches(np.array(fig.get_size_inches()) * mx * 4)
    plt.savefig(fname)


def aggr_imgs(log_dir, block, fname='aggregate_imgs.jpg'):
    data, basepath = get_img_data(log_dir, block)
    pairs = img_data_to_lst(data)
    save_images(os.path.join(basepath, fname), pairs[:2])


# def ed(x, y, x2, y2):
#     return np.sqrt((x - x2)**2 + (y - y2)**2)


# def color_gdist(x, y, lab, x2, y2, lab2):
#     return min(ed(x, y, x2, y2) + ciede2000(lab, lab2), 20.)


# def sift_gdist(x, y, o, x2, y2, o2, m=8.):
#     odiff = np.abs(o - o2)
#     return min(ed(x, y, x2, y2) + min(odiff, m - odiff), 2.)


# def oneto2d(index, side_len):
#     return index % side_len, int(index) / int(side_len)


# def color_gdist_mat(img, img2, side_len=32):
#     # assume imgs are the same size, square, and flattened
#     n = side_len**2
#     gdist = np.zeros([n, n])
#     for i in range(n):
#         x, y = oneto2d(i, side_len)
#         for j in range(n):
#             x2, y2 = oneto2d(j, side_len)
#             gdist[i, j] = color_gdist(x, y, img[i], x2, y2, img2[j])
#     return gdist

# def sift_gdist_mat(k, desc, k2, desc2, mod=128):
#     ln = desc.shape[0]
#     ln2 = desc2.shape[0]
#     gdist = np.zeros([ln, ln2])
#     for i in range(ln):
#         x, y = k[i / mod]
#         for j in range(ln2):
#             x2, y2 = k2[j / mod]
#             gdist[i, j] = sift_gdist(x, y, desc[i], x2, y2, desc2[j])
#     return gdist


# def make_indices(size):
#     o = []
#     for i in range(size):
#         for j in range(size):
#             for k in range(size):
#                 o.append([i, j, k])
#     return np.array(o)


# def compute_ciede2000_dict(quantize=8):
#     size = 256 / quantize
#     o = make_indices(size)
#     lab = np.squeeze(rgb2lab(np.expand_dims(np.array(o*quantize, np.float32) / 255., axis=0)))
#     n = o.shape[0]
#     dist = np.ones([n, n], np.float32)
#     for i in range(n):
#         for j in range(n):
#             dist[i, j] = ciede2000(lab[i], lab[j])
#     return o, dist


def flatten2color(im):
    return np.stack([im[:, :, i].flatten() for i in range(3)], axis=1)


def compute_sift(img, sift):
    im = img_as_ubyte(img)
    kp, des = sift.detectAndCompute(im, None)
    return kp, des


def pos_dist_mat(size):
    r = range(size)
    xx, yy = np.meshgrid(r, r)
    xy = np.stack([xx.flatten(), yy.flatten()], axis=1)
    dist = distance_matrix(xy, xy)
    yield
    while True:
        yield dist

def color_dist_mat(img, img2):
    im = flatten2color(img)
    im2 = flatten2color(img2)
    return distance_matrix(im, im2)


def img2color_dist_mat(img, img2, pdist):
    cdist = color_dist_mat(img, img2)
    return np.minimum(cdist + pdist, 20)


def sift_dist_sq(bins):
    r = range(4)
    xx, yy = np.meshgrid(r, r)
    xy = np.stack([xx.repeat(8, axis=1).flatten(), yy.repeat(8, axis=1).flatten()], axis=1)
    dist = distance_matrix(xy, xy)

    nbin_sets = 4*4
    r = range(bins)
    d = np.expand_dims(np.tile(r, [1, nbin_sets]).flatten(), axis=1)
    d2 = np.expand_dims(np.tile(r, [1, nbin_sets]).flatten(), axis=1)
    odiff = distance_matrix(d, d2, p=1)
    odiff = np.minimum(odiff, bins - odiff)
    dist = np.minimum(dist + odiff, 2)
    shape = yield
    while True:
        shape = yield np.tile(dist, shape)


# def binpos_dist_mat(desc, desc2):
#     lnd = desc.shape[0]
#     lnd2 = desc2.shape[0]
#     r = range(4)
#     xx, yy = np.meshgrid(r, r)
#     xy = np.stack([xx.repeat(8, axis=1).flatten(), yy.repeat(8, axis=1).flatten()], axis=1)
#     d = np.tile(xy, [lnd, 1])
#     d2 = np.tile(xy, [lnd2, 1])
#     return distance_matrix(d, d2)
#     k = np.array([p.pt for p in kp])
#     k2 = np.array([p.pt for p in kp2])
#     k = np.tile(k, cnt).flatten().reshape([lnk*cnt, 2])
#     k2 = np.tile(k2, cnt).flatten().reshape([lnk2*cnt, 2])
#     return distance_matrix(k, k2) #, threshold=(128*8)**2)


# def binorient_dist_mat(desc, desc2, bins):
#     nbin_sets = 4*4
#     lnd = desc.shape[0]
#     lnd2 = desc2.shape[0]
#     r = range(bins)
#     d = np.expand_dims(np.tile(r, [lnd, nbin_sets]).flatten(), axis=1)
#     d2 = np.expand_dims(np.tile(r, [lnd2, nbin_sets]).flatten(), axis=1)
#     odiff = distance_matrix(d, d2, p=1)
#     return np.minimum(odiff, bins - odiff)


# def descriptors_dist_mat(desc, desc2, bins):
#     d = np.expand_dims(desc.flatten(), axis=1)
#     d2 = np.expand_dims(desc2.flatten(), axis=1)
#     odiff = distance_matrix(d, d2, p=1)
#     return np.minimum(odiff, bins - odiff)


# def sift_dist_mat(kp, desc, kp2, desc2, cnt=4*4*8, bins=8):
#     pdist = binpos_dist_mat(desc, desc2)
#     odist = binorient_dist_mat(desc, desc2, bins)
#     return np.minimum(pdist + dist, 2)


def img2sift_desc(img, img2, sift):
    _, d = compute_sift(img, sift)
    _, d2 = compute_sift(img2, sift)
    return d, d2


def square_gdist(p, q, gdist):
    mxd = np.max(gdist)
    qsz = q.size
    psz = p.size
    if psz < qsz:
        add_n = qsz - psz
        pp = np.concatenate([p, np.zeros([add_n,])])
        qq = q
        dd = np.ones([qsz, qsz])*mxd # np.concatenate([gdist, np.ones([h, add_n])*mxd], axis=1)
        dd[:psz, :qsz] = gdist
    else:
        add_n = psz - qsz
        pp = p
        qq = np.concatenate([q, np.zeros([add_n,])])
        dd = np.ones([psz, psz])*mxd # np.concatenate([gdist, np.ones([add_n, w])*mxd], axis=0)
        dd[:psz, :qsz] = gdist
    return pp, qq, dd


def diag_expand_mat(mat): # square first
    sz = mat.shape[0]
    m = np.zeros([sz*2, sz*2])
    m[sz:, :sz] = mat
    m[:sz, sz:] = mat
    return m


def compute_emd(img, img2, sift_sq, pdist, sift):
    if sift_sq is not None:
        p, q = img2sift_desc(img, img2, sift)
        gdist = sift_sq.send([p.shape[0], q.shape[0]])
        # print p.shape, q.shape
        p = p.flatten()
        q = q.flatten()
        psz, qsz = gdist.shape
    elif pdist is not None:
        gdist = img2color_dist_mat(img, img2, pdist.next())
        psz, qsz = gdist.shape
        p = np.ones([psz])
        q = np.ones([qsz])
    else:
        raise TypeError('sift_sq or pdist must not be None.')
    if psz != qsz:
        p, q, gdist = square_gdist(p, q, gdist)
        psz, qsz = gdist.shape
    p = np.concatenate([p, np.zeros([qsz])])
    q = np.concatenate([np.zeros([psz]), q])
    gdist = diag_expand_mat(gdist)
    return emd(p, q, gdist)


def split_imgs(log_dir):
    splits = dict([(str(i), []) for i in range(10)])
    files = os.listdir(log_dir)
    for f in files:
        splits[f.split('_')[0]].append(f)
    return splits


def load_img(log_dir, f, grey, size):
    im = Image.open(os.path.join(log_dir, f))
    if grey:
        im = im.convert('L')
    if size is not None:
        im = im.resize(size, Image.BICUBIC)
    im = np.array(im, np.uint8)
    return im


def load_imgs(log_dir, files, grey=False, size=None):
    im = load_img(log_dir, files[0], grey, size)
    shape = [len(files)]+list(im.shape)
    out = np.zeros(shape, np.uint8)
    out[0] = im
    for i, f in enumerate(files[1:]):
        im = load_img(log_dir, f, grey, size)
        out[i+1] = im
    return out


def latent_to_one(ref_img, ref_latent, files, imgs, latents):
    dists = []
    cnt = imgs.shape[0]
    ref = np.tile(ref_latent, [latents.shape[0], 1])
    dist = np.linalg.norm(ref-latents, axis=1)
    for i in range(cnt):
        d = dist[i]
        dists.append((d, files[i], imgs[i]))
    return dists


def emd_to_one(ref_img, gref_img, files, imgs, gimgs, sift_sq, pdist, alpha, sift):
    dists = []
    cnt = imgs.shape[0]
    print_check = dict([(int(0.1*i*cnt), 0.1*i) for i in range(1, 10)])
    for i in range(cnt):
        # print type(gref_img), files[i]
        try:
            perc = print_check[i]
        except KeyError:
            pass
        else:
            print '{}% complete.'.format(perc)
        if alpha is not None:
            try:
                sdist = compute_emd(gref_img, gimgs[i], sift_sq, None, sift)
            except AttributeError:
                print files[i]
                continue
            cdist = compute_emd(ref_img, imgs[i], None, pdist, sift)
            dist = alpha*sdist + (1.-alpha)*cdist
        else:
            dist = compute_emd(ref_img, imgs[i], sift_sq, pdist, sift)
        dists.append((dist, files[i], imgs[i]))
    return dists


def emd_from_center(files, imgs, gimgs, sift_sq, pdist, alpha, sift):
    avg = np.array(np.mean(imgs, axis=0), np.uint8)
    if gimgs is not None:
        gavg = np.array(np.mean(gimgs, axis=0), np.uint8)
    else:
        gavg = None
    dists = emd_to_one(avg, gavg, files, imgs, gimgs, sift_sq, pdist, alpha, sift)
    # _, cfile, center = sorted(dists)[0]
    # dists = emd_to_one(center, files, imgs, sift_sq, pdist, alpha)
    return avg, dists


def euclid_center(imgs, files):
    cnt = imgs.shape[0]
    tile = [1 for _ in range(imgs.ndim)]
    tile[0] = cnt
    dists = []
    for i in range(cnt):
        im = np.tile(imgs[i], tile)
        d = np.mean(np.sqrt(np.sum((im - imgs)**2, axis=0)))
        dists.append((d, files[i]))
    return dists


def compute_center_dists(splits, log_dir, out_dir):
    centers = []
    for k in splits.keys():
        print 'Computing split {}.'.format(k)
        files = splits[k]
        imgs = load_imgs(log_dir, files, False, None)
        cdists = euclid_center(rgb2lab(imgs), files)
        cdists = sorted(cdists)
        centers.append((k, cdists[0][1]))
        with open(os.path.join(out_dir, '{}_cdists.pkl'.format(k)), 'w') as f:
            pickle.dump(cdists, f)
    with open(os.path.join(out_dir, 'all_cdists.pkl'), 'w') as f:
        pickle.dump(centers, f)


def get_points(h, w):
    y = range(1, h+1)
    x = range(1, w+1)
    xx, yy = np.meshgrid(x, y)
    xy = np.stack([xx.flatten(), yy.flatten()], axis=1)
    sz = xy.shape[0]
    xy = [xy[i] for i in range(sz)]
    return xy


def pt2index(pt):
    return tuple(np.int16(pt).tolist())


def nxt_imdist(k, imdists):
    dist, name, im = imdists.pop(0)
    return dist, name, im, k


def get_cluster_data(centers, alldists, points):
    clusters = dict([(int(k), {}) for k in alldists])
    c_pts = []
    next_imdists = []
    for k in clusters:
        data = clusters[k]
        data['c_img'], imdists = alldists[str(k)]
        data['imdists'] = sorted(imdists, key=lambda v: v[0])
        next_imdists.append(nxt_imdist(k, data['imdists']))
        _, cdists = get_center_dists(points, centers[k])
        data['c_pt'] = centers[k]
        c_pts.append(pt2index(centers[k]))
        data['ptdists'] = OrderedDict([(pt2index(pt), delta) for pt, delta in cdists])
    for cpt in c_pts:
        for data in clusters.values():
            del data['ptdists'][cpt]
    return clusters, sorted(next_imdists, key=lambda v: v[0])


def get_center_dists(xy, center=None):
    sz = len(xy)
    c_i = sz/2 - 1 - (xy[-1][1]/2)
    if center is None:
        center = xy[c_i]
    dists = sorted([(pt, np.sum((pt - center)**2)**0.5) for pt in xy], key=lambda v: v[1])
    return center, dists


def add_img_bw(canvas, img, pt):
    posh, posw = pt
    szh, szw = img.shape
    szh = szh/2
    szw = szw/2
    canvas[(posh-szh):posh+szh, (posw-szw):posw+szw] = img
    return canvas


def add_img(canvas, img, pt):
    posh, posw = pt
    szh, szw, c = img.shape
    szh = szh/2
    szw = szw/2
    canvas[(posh-szh):posh+szh, (posw-szw):posw+szw, :] = img
    return canvas


def get_center_tsne(all_dists, num_lbls, sift_sq, pdist, sift):
    centers = [all_dists[str(i)][0] for i in range(num_lbls)]
    gcenters = [np.array(Image.fromarray(centers[i]).convert('L')) for i in range(num_lbls)]
    dists = np.zeros([num_lbls, num_lbls])
    for i in range(num_lbls):
        print i
        for j in range(num_lbls):
            if i == j:
                break
            sd = compute_emd(gcenters[i], gcenters[j], sift_sq, None, sift)
            cd = compute_emd(centers[i], centers[j], None, pdist, sift)
            d = alpha*sd + (1.-alpha)*cd
            dists[i, j] = d
            dists[j, i] = d
    dists_2d = TSNE(n_components=2, metric='precomputed').fit_transform(dists)
    dists_2d -= np.min(dists_2d)
    dists_2d /= np.max(dists_2d)
    return dists_2d

# def calc_emds(files, imgs, sift_sq, pdist):
#     if sift_sq is not None and pdist is not None:
#         raise TypeError('Only one of sift_sq and pdist can be not None.')
#     cnt = imgs.shape[0]
#     dists = {}
#     print_check = dict([(int(0.1*i*cnt), 0.1*i) for i in range(1, 10)])
#     for i in range(cnt):
#         try:
#             perc = print_check[i]
#         except KeyError:
#             pass
#         else:
#             print perc
#         for j in range(cnt):
#             if i == j:
#                 break
#             print files[i], files[j]
#             dists[(files[i], files[j])] = compute_emd(imgs[i], imgs[j], None, sift_sq, pdist)
#     return dists


def split_celeb(fname='/home/olias/data/list_attr_celeba.txt'):
    with open(fname, 'r') as f:
        cnt = f.readline().rstrip()
        labels = f.readline().rstrip().split()
        data = f.readlines()
    splits = dict([(i, []) for i in range(len(labels))])
    for row in data:
        rw = row.rstrip().split()
        img = rw[0]
        vals = rw[1:]
        for i, v in enumerate(vals):
            if v == '1':
                splits[i].append(img)
    return labels, splits


def split_gender_and_filter(splits, lbls, testonly=162770):
    if testonly is not None:
        for k in splits:
            splits[k] = [im for im in splits[k] if int(im.split('.')[0]) > testonly]
    _all = set([])
    for k in splits:
        _all |= set(splits[k])
    man = dict([(m, None) for m in splits[20]])
    women = list(_all - set(man.keys()))
    msplits = dict([(k, []) for k in splits])
    wsplits = dict([(k, []) for k in splits])
    for k in splits:
        for im in splits[k]:
            try:
                _ = man[im]
            except KeyError:
                wsplits[k].append(im)
            else:
                msplits[k].append(im)
    # filter out labels that are either too representative or too rare
    for k in splits:
        mv = len(msplits[k]) / float(len(man.keys()))
        wv = len(wsplits[k]) / float(len(women))
        if mv > 0.4 or mv < 0.01:
            del msplits[k]
        if wv > 0.4 or wv < 0.009: # accounts for rounding
            del wsplits[k]
    return msplits, wsplits


def to_paper_grps(splits, labels):
    grouping = dict([(1, ['Mouth_Slightly_Open',
                          'High_Cheekbones',
                          'Smiling']),
                     (2, ['Attractive',
                          'No_Beard',
                          'Heavy_Makeup',
                          'Young',
                          'Wavy_Hair',
                          'Bangs',
                          'Wearing_Lipstick',
                          'Brown_Hair',
                          'Pointy_Nose',
                          'Rosy_Cheeks',
                          'Oval_Face']),
                     (3, ['Blond_Hair',
                          'Gray_Hair',
                          'Pale_Skin',
                          'Blurry']),
                     (4, ['Black_Hair',
                          'Straight_Hair',
                          'Wearing_Hat',
                          'Eyeglasses']),
                     (5, ['Wearing_Necktie',
                          'Male',
                          'Bald',
                          '5_o_Clock_Shadow',
                          'Sideburns',
                          'Mustache',
                          'Goatee']),
                     (6, ['Wearing_Earrings',
                          'Wearing_Necklace',
                          'Chubby',
                          'Double_Chin',
                          'Receding_Hairline',
                          'Arched_Eyebrows',
                          'Bushy_Eyebrows',
                          'Narrow_Eyes',
                          'Bags_Under_Eyes',
                          'Big_Nose',
                          'Big_Lips'])])
    lbl_i = dict([(lbl, i) for i, lbl in enumerate(labels)])
    grps = dict([(k, set([])) for k in grouping])
    for k, lbls in grouping.items():
        for lbl in lbls:
            grps[k] |= set(splits[lbl_i[lbl]])
    for k in grps:
        grps[k] = list(grps[k])
    return grps
