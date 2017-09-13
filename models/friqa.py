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
            tf.add(num, tf.reduce_sum(log10(tf.clip_by_value(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq), 1e-10, 1e10)))), \
            tf.add(den, tf.reduce_sum(log10(tf.clip_by_value(1 + sigma1_sq / sigma_nsq, 1e-10, 1e10)))), \
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
    #num = tf.Print(output[3], [output[3], output[4]], 'Num and Den values: ', -1)
    return output[3]/output[4]


def prep_and_call_vifp(ref, dist):
    out = vifp(tograyscale((ref+1)/2)*255, tograyscale((dist+1)/2)*255)
    out = tf.cond(out < tf.constant(0.0), lambda: tf.constant(0.0), lambda: out)
    out = tf.cond(out > tf.constant(1.0), lambda: tf.constant(1.0), lambda: out)
    return 1. - out


def vifp_mscale(ref, dist):
    #From: https://github.com/aizvorski/video-quality
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


def toyiq(img):
    r, g, b = tf.unstack(img, num=3, axis=1)
    y = 0.30*r + 0.59*g + 0.11*b
    i = 0.60*r - 0.28*g - 0.32*b
    q = 0.21*r - 0.52*g + 0.31*b
    return y, i, q


def fromyiq(img):
    y, i, q = tf.unstack(img, num=3, axis=1)
    r = tf.clip_by_value(y + 0.948262*i + 0.624013*q, 0, 1.0)
    g = tf.clip_by_value(y - 0.276066*i - 0.639810*q, 0, 1.0)
    b = tf.clip_by_value(y - 1.105450*i + 1.729860*q, 0, 1.0)
    return r, g, b


def np_fromyiq(img):
    y = img[:, :, :, 0]
    i = img[:, :, :, 1]
    q = img[:, :, :, 2]
    img2 = img.copy()
    img2[:, :, :, 0] = np.clip(y + 0.948262*i + 0.624013*q, 0, 1.0)
    img2[:, :, :, 1] = np.clip(y - 0.276066*i - 0.639810*q, 0, 1.0)
    img2[:, :, :, 2] = np.clip(y - 1.105450*i + 1.729860*q, 0, 1.0)
    return img2

def sobel(yiq_img):
    #From: https://stackoverflow.com/questions/35565312/is-there-a-convolution-function-in-tensorflow-to-apply-a-sobel-filter
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

    G_x = tf.nn.conv2d(tf.expand_dims(yiq_img, 1), sobel_x_filter,
                       strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
    G_y = tf.nn.conv2d(tf.expand_dims(yiq_img, 1), sobel_y_filter,
                       strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')

    return G_x, G_y


def dice_metric(r, d, c):
    return (2*r*d + c)/(r**2 + d**2 + c)


def gms(ry, dy, c):
    rG_x, rG_y = sobel(ry)
    dG_x, dG_y = sobel(dy)

    rG = tf.sqrt(rG_x**2 + rG_y**2)
    dG = tf.sqrt(dG_x**2 + dG_y**2)

    return dice_metric(rG, dG, c)


def chrominance(ri, rq, di, dq, c):
    Ik = dice_metric(ri, di, c)
    Qk = dice_metric(rq, dq, c)
    return Ik*Qk


def qs(ref, dist):
    c = tf.constant(0.0026)
    ry, ri, rq = toyiq(ref)
    dy, di, dq = toyiq(dist)
    #sy, si, sq = tf.shape(ry), tf.shape(ri), tf.shape(rq)
    #ry = tf.Print(ry, [sy, si, sq])

    gms_ = gms(ry, dy, c)
    chrome = chrominance(ri, rq, di, dq, c)

    #out = tf.reduce_mean(gms_ + chrome)

    return tf.reduce_mean(gms_), tf.reduce_mean(chrome)#tf.Print(out, [gms_, chrome, out])


def qs_yiq(ref, dist):
    c = tf.constant(0.0026)
    ry, ri, rq = tf.unstack(ref, num=3, axis=1)
    dy, di, dq = tf.unstack(dist, num=3, axis=1)

    gms_ = gms(ry, dy, c)
    chrome = chrominance(ri, rq, di, dq, c)

    return tf.reduce_mean(gms_), tf.reduce_mean(chrome)#tf.Print(out, [gms_, chrome, out])


def prep_and_call_qs(ref, dist):
    gms_, chrome = qs(((ref+1)/2)*255., ((dist+1)/2)*255.)
    #out = tf.Print(out, [out])
    return 1.-gms_, 1.-chrome#(2. - out)/2.
