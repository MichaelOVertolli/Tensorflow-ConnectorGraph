from preprocessing import *


BASE_DIR = '/home/olias/data'
CELEB_DIR = os.path.join(BASE_DIR, 'img_align_celeba')


def setup(mnist=None, celeb=None, cifar10=None):
    types = ['_train', '_test', '_all']
    
    #mnist setup
    if mnist is not None:
        print 'Starting MNIST setup.'
        mtrain, mtest, mall = [mnist+typ for typ in types]
        mshape = [32, 32, 3]
        mnist_train = os.path.join(BASE_DIR, mtrain)
        mnist_test = os.path.join(BASE_DIR, mtest)
        mnist_all = os.path.join(BASE_DIR, mall)
        mnist_to_imgs(mnist_train, base_size=32, train=True, all_dir=mnist_all)
        mnist_to_imgs(mnist_test, base_size=32, train=False, all_dir=mnist_all)
        print 'Finished saving images.'
        print 'Starting tfrecords.'
        to_tfrecord_shard(mtrain, mnist_train+'r', mnist_train, mshape)
        print 'Finished sharding.'
        to_tfrecord(mtest, mnist_test+'r', mnist_test, mshape, shuffle=False)
        to_tfrecord(mall, mnist_all+'r', mnist_all, mshape, shuffle=False)
        print 'Finished all tfrecords.'

    if celeb is not None:
        print 'Starting CelebA setup.'
        cbtrain, cbtest, cball = [celeb+typ for typ in types]
        cbshape = [128, 128, 3]
        celeb_train = os.path.join(BASE_DIR, cbtrain)
        celeb_test = os.path.join(BASE_DIR, cbtest)
        celeb_all = os.path.join(BASE_DIR, cball)
        celeb_to_imgs(celeb_train, imgs_dir=CELEB_DIR, train=True, all_dir=celeb_all)
        celeb_to_imgs(celeb_test, imgs_dir=CELEB_DIR, train=False, all_dir=celeb_all)
        print 'Finished saving images.'
        print 'Starting tfrecords.'
        to_tfrecord_shard(cbtrain, celeb_train+'r', celeb_train, cbshape)
        print 'Finished sharding.'
        to_tfrecord(cbtest, celeb_test+'r', celeb_test, cbshape, shuffle=False)
        to_tfrecord(cball, celeb_all+'r', celeb_all, cbshape, shuffle=False)
        print 'Finished all tfrecords.'

    if cifar10 is not None:
        print 'Starting Cifar10 setup.'
        citrain, citest, ciall = [cifar10+typ for typ in types]
        cishape = [32, 32, 3]
        cifar10_train = os.path.join(BASE_DIR, citrain)
        cifar10_test = os.path.join(BASE_DIR, citest)
        cifar10_all = os.path.join(BASE_DIR, ciall)
        cifar10_to_imgs(cifar10_train, train=True, all_dir=cifar10_all)
        cifar10_to_imgs(cifar10_test, train=False, all_dir=cifar10_all)
        print 'Finished saving images.'
        print 'Starting tfrecords.'
        to_tfrecord_shard(citrain, cifar10_train+'r', cifar10_train, cishape)
        print 'Finished sharding.'
        to_tfrecord(citest, cifar10_test+'r', cifar10_test, cishape, shuffle=False)
        to_tfrecord(ciall, cifar10_all+'r', cifar10_all, cishape, shuffle=False)
        print 'Finished all tfrecords.'
