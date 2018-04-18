from models.subgraph import *
import numpy as np
from runner import *
import tensorflow as tf


def expt_CelebA_degraded():
    with tf.Session(graph=tf.Graph()) as sess:
        g=BuiltSubGraph('cqs_generator_loop_0', 'z128_sz64', sess, './logs/cqs_cg_full_reverse_scaled_began_gmsm_b16_z128_sz64_g0.7_1101_052228')
        fg = g.freeze(sess)
    r = Runner([fg], 'experiments/reverse/expt_CelebA_degraded_sz64')
    real = r.get_image_from_loader()
    random = np.random.uniform(-1, 1, [16, 3, 64, 64])
    z = np.random.uniform(-1, 1, [16, 128])
    z2 = np.random.uniform(-1, 1, [16, 128])
    fake = r.sess.run(fg.output_names[0], {fg.input_names[0]: z})
    fake2 = r.sess.run(fg.output_names[0], {fg.input_names[0]: z2})
    distorted = np.copy(fake)
    distorted[:, :, :20, :] = 0
    blend = fake*0.5 + fake2*0.5
    real_out = r.sess.run(fg.output_names[2], {fg.input_names[1]: real})
    real_out2 = r.sess.run(fg.output_names[2], {fg.input_names[1]: real_out})
    distorted_out = r.sess.run(fg.output_names[2], {fg.input_names[1]: distorted})
    distorted_out2 = r.sess.run(fg.output_names[2], {fg.input_names[1]: distorted_out})
    random_out = r.sess.run(fg.output_names[2], {fg.input_names[1]: random})
    random_out2 = r.sess.run(fg.output_names[2], {fg.input_names[1]: random_out})
    random_out3 = r.sess.run(fg.output_names[2], {fg.input_names[1]: random_out2})
    blend_out = r.sess.run(fg.output_names[2], {fg.input_names[1]: blend})
    blend_out2 = r.sess.run(fg.output_names[2], {fg.input_names[1]: blend_out})
    r.save_img(real, 'real_in.png')
    r.save_img(fake, 'fake_in.png')
    r.save_img(distorted, 'distorted_in.png')
    r.save_img(real_out, 'real_out.png')
    r.save_img(real_out2, 'real_out2.png')
    r.save_img(distorted_out, 'distorted_out.png')
    r.save_img(distorted_out2, 'distorted_out2.png')
    r.save_img(random, 'random.png')
    r.save_img(random_out, 'random_out.png')
    r.save_img(random_out2, 'random_out2.png')
    r.save_img(random_out3, 'random_out3.png')
    r.save_img(random_out, 'random_out.png')
    r.save_img(real, 'real_in.png')
    r.save_img(fake, 'fake_in.png')
    r.save_img(distorted, 'distorted_in.png')
    r.save_img(real_out, 'real_out.png')
    r.save_img(real_out2, 'real_out2.png')
    r.save_img(distorted_out, 'distorted_out.png')
    r.save_img(distorted_out2, 'distorted_out2.png')
    r.save_img(blend, 'blend.png')
    r.save_img(blend_out, 'blend_out.png')
    r.save_img(blend_out2, 'blend_out2.png')


def expt_CelebA_compare():
    with tf.Session(graph=tf.Graph()) as sess:
        g=BuiltSubGraph('cqs_generator_loop_0', 'z128_sz64', sess, './logs/cqs_cg_full_reverse_scaled_began_gmsm_b16_z128_sz64_g0.7_1101_052228')
        fg = g.freeze(sess)
    with tf.Session(graph=tf.Graph()) as sess:
        g2=BuiltSubGraph('cqs_generator_0', 'z128_sz64', sess, './logs/cqs_cg_began_b16_z128_sz64_g0.7_1103_002922')
        fg2 = g2.freeze(sess)
    r = Runner([fg, fg2], 'experiments/reverse/expt_CelebA_compare_sz64')
    real = r.get_image_from_loader()
    z = np.random.uniform(-1, 1, [16, 128])
    fake_scaled = r.sess.run(fg.output_names[0], {fg.input_names[0]: z})
    real_out_scaled = r.sess.run(fg.output_names[2], {fg.input_names[1]: real})
    real_out2_scaled = r.sess.run(fg.output_names[2], {fg.input_names[1]: real_out})
    fake_began = r.sess.run(fg2.output_names[0], {fg.input_names[0]: z})
    real_out_began = r.sess.run(fg2.output_names[2], {fg.input_names[1]: real})
    real_out2_began = r.sess.run(fg2.output_names[2], {fg.input_names[1]: real_out})
    r.save_img(real, 'real_in.png')
