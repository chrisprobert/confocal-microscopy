import os
import glob
import random

import numpy as np
import skimage
from skimage import exposure

import matplotlib.pylab as plt

import .evaluate_results


data_base_dir = os.path.join(
    '/Users/chris/data/', 'flmicroscopy/rawdata/experiments/')
exp_dirs_regex = '{}*/*/*/YRC*'.format(data_base_dir)
exp_dirs = glob.glob(exp_dirs_regex)


def imgs_from_exp_dir(exp_dir):
    return glob.glob(exp_dir + '/images/*.tiff')

expdir2imgs = {d: imgs_from_exp_dir(d) for d in exp_dirs}
kvs = list(expdir2imgs.items())
for (k, v) in kvs:
    if len(v) == 0:
        del expdir2imgs[k]


def pltimgcbar(im):
    f = plt.figure(figsize=(6, 6))
    plt.imshow(im)
    plt.colorbar()
    plt.show()


def pltimg(im, show=True):
    f = plt.figure(figsize=(6, 6))
    plt.imshow(im)
    if show:
        plt.show()


def pltimgbw(im, show=True):
    f = plt.figure(figsize=(6, 6))
    plt.imshow(im, cmap='Greys')
    if show:
        plt.show()


def pixelhist(im, nbins=100):
    a = im.flatten()
    bins = np.linspace(a.min(), a.max(), nbins)
    f = plt.figure(figsize=(6, 6))
    plt.hist(a, bins=bins)
    plt.show()


def plithist(im, nbins=256):
    f = plt.figure(figsize=(6, 6))
    plt.hist(im.ravel(), bins=nbins, histtype='step', color='black')
    img_cdf, bins = exposure.cumulative_distribution(im, nbins)
    plt.plot(bins, img_cdf, 'r')
    plt.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    plt.xlabel('Pixel intensity')
    plt.xlim(0, 1)
    plt.yticks([])


def apply_pthresh(im, pthresh):
    aim = im.copy()
    thresh = np.percentile(aim, pthresh)
    aim[(aim < thresh)] = aim.min()
    return aim


def apply_pthresh_bw(im, pthresh):
    aim = apply_pthresh(im, pthresh)
    aim[(aim == aim.min())] = 0
    aim[(aim != 0)] = 1
    return aim


def loadim(impath):
    '''load image and rescale on 0-1'''
    im = plt.imread(impath)
    im = skimage.img_as_float(im)
    im = (im - im.min()) / (im.max() - im.min())
    return im

imgs = list(expdir2imgs.values())
random.shuffle(imgs)


def show_sample_images():
    for i, imglist in enumerate(imgs[:5]):
        for j, ifile in enumerate(imglist):
            print('{},{}'.format(i, j))
            im = loadim(ifile)
            plt.imshow(im)
            plt.show()


def plot_image_with_bboxes(im_to_plot, im_to_gen_bboxes):
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(im_to_plot)

    bboxes = evaluate_results.get_bboxes(im_to_gen_bboxes)

    for (b_xs, b_ys) in bboxes:
        Xs = [b_xs[0], b_xs[0], b_xs[1], b_xs[1], b_xs[0]]
        Ys = [b_ys[0], b_ys[1], b_ys[1], b_ys[0], b_ys[0]]
        line = plt.Line2D(Ys, Xs, color='red', linestyle='solid')
        fig.add_subplot(111).add_artist(line)

    plt.show()
