import tensorflow as tf
import numpy as np


def load_testimg():
    from PIL import Image
    return np.expand_dims(np.transpose(np.asarray(
        Image.open('AE_G19.png').convert('RGB'), np.float32))/256., 0)


def testsetup():
    img = load_testimg()
    timg = tf.placeholder(tf.float32, [1, 3, 64, 64])
    g = tograyscale(timg)*256.
    gaus = gaussian_filter_gray(g, tf.constant(0.5, tf.float32), 'NCHW')
    out = vifp(g, gaus)
    return img, timg, g, gaus, out


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def gaussian_filter(input, sigma, data_format):
    #assume that args are already tf values in graph
    lw = 4.0 * sigma + 0.5
    half = tf.exp(tf.square(tf.range(1, lw+1)) * -0.5 / sigma**2)
    gaussian = tf.concat([tf.reverse(half, [0]), [1.0], half], 0)
    gaussian = tf.tile([gaussian], [tf.shape(gaussian)[0], 1])
    gaussian2d = gaussian * tf.transpose(gaussian)
    gaussian2d = gaussian2d / tf.reduce_sum(gaussian2d)

    gaussian4d = tf.expand_dims(tf.expand_dims(gaussian2d, 2), 3)

    c1, c2, c3 = tf.unstack(input, num=3, axis=1)
    c1 = tf.nn.conv2d(tf.expand_dims(c1, 1), gaussian4d, [1, 1, 1, 1], 'SAME', data_format=data_format)
    c2 = tf.nn.conv2d(tf.expand_dims(c2, 1), gaussian4d, [1, 1, 1, 1], 'SAME', data_format=data_format)
    c3 = tf.nn.conv2d(tf.expand_dims(c3, 1), gaussian4d, [1, 1, 1, 1], 'SAME', data_format=data_format)
    return tf.concat([c1, c2, c3], 1)


def gaussian_filter_gray_alt(input, sigma, data_format):
    lw = 4.0*sigma+0.5
    ax = tf.range(-lw // 2 + 1., l // 2 + 1.)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.expand_dims(tf.expand_dims(gaussian, 2), 3)

    return tf.nn.conv2d(input, kernel, [1, 1, 1, 1], 'SAME', data_format=data_format)


def gaussian_filter_gray(input, sigma, data_format):
    #input MUST BE 0..256
    lw = 4.0 * sigma + 0.5
    half = tf.exp(tf.square(tf.range(1, lw+1)) * -0.5 / sigma**2)
    gaussian = tf.concat([tf.reverse(half, [0]), [1.0], half], 0)
    gaussian = tf.tile([gaussian], [tf.shape(gaussian)[0], 1])
    gaussian2d = gaussian * tf.transpose(gaussian)
    gaussian2d = gaussian2d / tf.reduce_sum(gaussian2d)

    gaussian4d = tf.expand_dims(tf.expand_dims(gaussian2d, 2), 3)
    shape = tf.shape(gaussian4d)
    h = (shape[0]-1)/2
    w = (shape[1]-1)/2
    padded = tf.pad(input, [[0, 0], [0, 0], [h, h], [w, w]], "REFLECT")

    return tf.nn.conv2d(padded, gaussian4d, [1, 1, 1, 1], 'VALID', data_format=data_format)


def tograyscale(img):
    r, g, b = tf.unstack(img, num=3, axis=1)
    c = 0.2126*r + 0.7152*g + 0.0722*b
    split = tf.constant(0.0031308, tf.float32)
    comp1 = tf.less_equal(c, split)
    return tf.expand_dims(tf.where(comp1, 12.92*c, 1.055*tf.pow(c, 1/2.4)-0.055), 1)


def scale_loop(scale, ref, dist, num, den, eps, sigma_nsq, zero):
    N = 2**(4-scale+1) + 1
    sigma = N/5.0

    mu1 = gaussian_filter_gray(ref, sigma, 'NCHW')
    mu2 = gaussian_filter_gray(dist, sigma, 'NCHW')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = gaussian_filter_gray(ref*ref, sigma, 'NCHW') - mu1_sq
    sigma2_sq = gaussian_filter_gray(dist*dist, sigma, 'NCHW') - mu2_sq
    sigma12 = gaussian_filter_gray(ref*dist, sigma, 'NCHW') - mu1_mu2

    zeros1 = tf.zeros_like(sigma1_sq)
    sigma1_sq = tf.where(tf.less(sigma1_sq, zero), zeros1, sigma1_sq)
    sigma2_sq = tf.where(tf.less(sigma2_sq, zero), tf.zeros_like(sigma2_sq), sigma2_sq)

    g = sigma12 / (sigma1_sq + eps)
    sv_sq = sigma2_sq - (g * sigma12)

    zerosg = tf.zeros_like(g)
    g = tf.where(tf.less(sigma1_sq, eps), zerosg, g)
    sv_sq = tf.where(tf.less(sigma1_sq, eps), sigma2_sq, sv_sq)
    sigma1_sq = tf.where(tf.less(sigma1_sq, eps), zeros1, sigma1_sq)

    g = tf.where(tf.less(sigma2_sq, eps), zerosg, g)
    sv_sq = tf.where(tf.less(sigma2_sq, eps), tf.zeros_like(sv_sq), sv_sq)

    sv_sq = tf.where(tf.less(g, zero), sigma2_sq, sv_sq)
    g = tf.where(tf.less(g, zero), zerosg, g)
    sv_sq = tf.where(tf.less_equal(sv_sq, eps), tf.ones_like(sv_sq)*eps, sv_sq)

    return [scale+1, \
            gaussian_filter_gray(ref, sigma, 'NCHW')[:, :, ::2, ::2], \
            gaussian_filter_gray(dist, sigma, 'NCHW')[:, :, ::2, ::2], \
            tf.add(num, tf.reduce_sum(log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))), \
            tf.add(den, tf.reduce_sum(log10(1 + sigma1_sq / sigma_nsq))), \
            eps, sigma_nsq, zero]


def vifp(ref, dist):
    sigma_nsq = tf.constant(2, tf.float32)
    eps = tf.constant(1e-10, tf.float32)
    zero = tf.constant(0, tf.float32)
    scale = tf.constant(1, tf.float32)

    cond = lambda scale_, ref_, dist_, num, den, eps_, sigma_nsq_, zero_: \
           tf.less(scale_, tf.constant(5, tf.float32))
    start_vals = [scale, \
                  ref, dist, \
                  zero, zero, \
                  eps, sigma_nsq, \
                  zero]
    shape = tf.TensorShape([None, 1, None, None])
    output = tf.while_loop(cond, scale_loop, start_vals,
                           shape_invariants=[scale.get_shape(), shape, shape, zero.get_shape(),
                                             zero.get_shape(), eps.get_shape(),
                                             sigma_nsq.get_shape(), zero.get_shape()])
    return output[3]/output[4]


def vifp_mscale(ref, dist):
    import numpy
    import scipy.signal
    import scipy.ndimage
    
    sigma_nsq=2
    eps = 1e-10

    num = 0.0
    den = 0.0
    for scale in range(1, 5):
       
        N = 2**(4-scale+1) + 1
        sd = N/5.0

        if (scale > 1):
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]
                
        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2
    
        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0
        
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        
        g[sigma1_sq<eps] = 0
        sv_sq[sigma1_sq<eps] = sigma2_sq[sigma1_sq<eps]
        sigma1_sq[sigma1_sq<eps] = 0
        
        g[sigma2_sq<eps] = 0
        sv_sq[sigma2_sq<eps] = 0
        
        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=eps] = eps
        
        num += numpy.sum(numpy.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += numpy.sum(numpy.log10(1 + sigma1_sq / sigma_nsq))
        
    vifp = num/den

    return vifp
