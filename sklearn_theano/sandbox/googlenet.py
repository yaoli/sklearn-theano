'''
Demonstrate how to build a classifier using GoogleNet.

The default setup runs on the *validset* of ImageNet.

Classification accuracy: TOP1 69.02%, TOP5 89.08%

The current processing takes only one center crop of input images.
In the GoogleNet paper, they use more aggressive cropping to acheive
TOP5 93.33% on the *testset*. For details, please refer to the
orignal paper.

Disclaim: some of the utility functions are borrowed directly from Caffe.
'''
import glob
import re
import time

from sklearn_theano.feature_extraction.caffe.googlenet \
  import create_theano_expressions
import skimage
import skimage.io
import skimage.transform
from skimage.transform import resize
import numpy

import theano

# change these paths according to your specific setup
IMAGE = '/data/lisatmp3/yaoli/datasets/ILSVRC2012/ILSVRC2012/valid/'
LABELS = '/data/lisatmp3/yaoli/caffe/caffe/data/ilsvrc12/val.txt'
MEAN = '/data/lisatmp3/yaoli/caffe/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
EXT = 'JPEG'

def generate_minibatch_idx(dataset_size, minibatch_size):
    # generate idx for minibatches
    # output [m1, m2, m3, ..., mk] where mk is a list of indices
    assert dataset_size >= minibatch_size
    n_minibatches = dataset_size / minibatch_size
    leftover = dataset_size % minibatch_size
    idx = range(dataset_size)
    if leftover == 0:
        minibatch_idx = numpy.split(numpy.asarray(idx), n_minibatches)
    else:
        print 'uneven minibath chunking, overall %d, last one %d'%(minibatch_size, leftover)
        minibatch_idx = numpy.split(numpy.asarray(idx)[:-leftover], n_minibatches)
        minibatch_idx = minibatch_idx + [numpy.asarray(idx[-leftover:])]
    minibatch_idx = [idx_.tolist() for idx_ in minibatch_idx]
    return minibatch_idx

def load_txt_file(path):
    f = open(path,'r')
    lines = f.readlines()
    f.close()
    return lines

def sort_by_numbers_in_file_name(list_of_file_names):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('(\-?[0-9]+)', s) ]

    def sort_nicely(l):
        """ Sort the given list in the way that humans expect.
        """
        l.sort(key=alphanum_key)
        return l
    return sort_nicely(list_of_file_names)

def load_image(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.

    Take
    filename: string
    color: flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).

    Give
    image: an image with type numpy.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    img = skimage.img_as_float(skimage.io.imread(filename)).astype(numpy.float32)
    if img.ndim == 2:
        img = img[:, :, numpy.newaxis]
        if color:
            img = numpy.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def oversample_image(images, crop_dims):
    """
    Crop images into the four corners, center, and their mirrored versions.

    Take
    image: iterable of (H x W x K) ndarrays
    crop_dims: (height, width) tuple for the crops.

    Give
    crops: (10*N x H x W x K) ndarray of crops for number of inputs N.
    """
    # Dimensions and center.
    im_shape = numpy.array(images[0].shape)
    crop_dims = numpy.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = numpy.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = numpy.tile(im_center, (1, 2)) + numpy.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = numpy.tile(crops_ix, (2, 1))

    # Extract crops
    crops = numpy.empty((10 * len(images), crop_dims[0], crop_dims[1],
                            im_shape[-1]), dtype=numpy.float32)
    ix = 0
    for im in images:
        for crop in crops_ix:
            crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
            ix += 1
        crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]  # flip for mirrors
    return crops

def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.

    Take
    im: (H x W x K) ndarray
    new_dims: (height, width) tuple of new dimensions.
    interp_order: interpolation order, default is linear.

    Give
    im: resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        # skimage is fast but only understands {1,3} channel images in [0, 1].
        im_min, im_max = im.min(), im.max()
        im_std = (im - im_min) / (im_max - im_min)
        resized_std = resize(im_std, new_dims, order=interp_order)
        resized_im = resized_std * (im_max - im_min) + im_min
    else:
        # ndimage interpolates anything but more slowly.
        # but this handles batch
        scale = tuple(numpy.array(new_dims) / (numpy.array(im.shape[:2])+.0))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(numpy.float32)

class GoogleNet(object):
    def __init__(self):
        print 'build GoogleNet from sklearn-theano'
        self.image_dims = [256, 256]
        self.channel_swap = [2, 1, 0]
        self.raw_scale = 255.0
        self.crop_dims = numpy.array([224, 224])
        self.mean = self.set_mean()
        
    def fprop(self):
        expression, inputs = create_theano_expressions()
        outputs = expression['loss3/loss3']
        return inputs, outputs

    def build_model(self):
        print 'compile GoogleNet'
        t0 = time.time()
        x, probs = self.fprop()
        self.classify_fn = theano.function([x], probs)
        print 'took ', time.time() - t0
        
    def set_mean(self, mode='elementwise'):
        """
        Set the mean to subtract for data centering.

        Take
        mean: mean K x H x W ndarray (input dimensional or broadcastable)
        mode: elementwise = use the whole mean (and check dimensions)
              channel = channel constant (e.g. mean pixel instead of mean image)
        """
        mean = numpy.load(MEAN)
        crop_dims = tuple(self.crop_dims.tolist())
        if mode == 'elementwise':
            if mean.shape[1:] != crop_dims:
                # Resize mean (which requires H x W x K input).
                mean = resize_image(mean.transpose((1,2,0)),
                                    crop_dims).transpose((2,0,1))
        elif mode == 'channel':
            mean = mean.mean(1).mean(1).reshape((in_shape[1], 1, 1))
        elif mode == 'nothing':
            mean = mean.mean(0)
        else:
            raise Exception('Mode not in {}'.format(['elementwise', 'channel']))
        return mean
    
    def preprocess(self, inputs, oversample=False):
        """
        inputs: iterable of (H x W x K) input ndarrays
        oversample: average predictions across center, corners, and mirrors
                    when True (default). Center-only prediction when False.
        """
        
        # Scale to standardize input dimensions.
        input_ = numpy.zeros((len(inputs),
            self.image_dims[0], self.image_dims[1], inputs[0].shape[2]),
            dtype=numpy.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = resize_image(in_, self.image_dims)
        
        if oversample:
            # Generate center, corner, and mirrored crops.
            input_ = oversample_image(input_, self.crop_dims)
        else:
            # Take center crop.
            center = numpy.array(self.image_dims) / 2.0
            crop = numpy.tile(center, (1, 2))[0] + numpy.concatenate([
                -self.crop_dims / 2.0,
                self.crop_dims / 2.0
            ])
            input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]
        
        # Classify
        ins = numpy.zeros(numpy.array(input_.shape)[[0,3,1,2]],
                            dtype=numpy.float32)
        def transform(x):
            x_ = x.astype(numpy.float32, copy=False) # (224,224,3)
            x_ = x_[:, :, self.channel_swap]
            x_ = x_.transpose((2, 0, 1))
            x_ *= self.raw_scale
            x_ -= self.mean # mean is between 0 and 255
            return x_
        for ix, in_ in enumerate(input_):
            ins[ix] = transform(in_)
        return ins    
            
    def classify(self):
        files = glob.glob(IMAGE + '/*.' + EXT)
        files = sort_by_numbers_in_file_name(files)
        labels = load_txt_file(LABELS)
        labels = [int((label.split(' ')[-1]).strip()) for label in labels]
        # go through minibatches
        idx = generate_minibatch_idx(len(files), 100)
     
        TOP1s = []
        TOP5s = []
        for i, index in enumerate(idx):
            current = [files[j] for j in index]
            gts = numpy.asarray([labels[j] for j in index])
            inputs =[load_image(im_f) for im_f in current]
            inputs = self.preprocess(inputs)
            probs = self.classify_fn(inputs) # (m, 1000, 1, 1)
            probs = numpy.squeeze(probs)
            predictions = probs.argsort()[:, ::-1][:, :5]
            for pred, gt in zip(predictions, gts):
                TOP1 = pred[0] == gt
                TOP5 = gt in pred
                TOP1s.append(TOP1)
                TOP5s.append(TOP5)
            print '%d / %d minibatches, acu TOP1 %.4f, TOP5 %.4f'%(
                i, len(idx), numpy.mean(TOP1s) * 100, numpy.mean(TOP5s) * 100)
                
def test_classifier():
    model = GoogleNet()
    model.build_model()
    model.classify()
    
if __name__ == '__main__':
    test_classifier()
