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

from models.friqa import prep_and_call_qs_sim
from models.model_utils import *
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
