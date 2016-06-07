import numpy as np


class EdgeExtender(object):
    '''Base edge extender class'''

    def __init__(self, name):
        self.name = name

    def apply(self, im, dims):
        raise NotImplementedError('Not implemented in base class')


class PerpendicularEdgeExtender(EdgeExtender):
    '''Extend image using standard perpendicular extension'''

    def __init__(self):
        super(PerpendicularEdgeExtender, self).__init__(
            'PerpendicularEdgeExtender')

    def apply(self, im, dims):
        new_im = np.zeros(np.array(im.shape) + 2 * np.array(dims))
        new_im[dims[0]:dims[0] + im.shape[0], dims[1]
            :dims[1] + im.shape[1]] = im[:, :]
        new_im[0:dims[0], 0:dims[1]] = im[0, 0]
        new_im[dims[0] + im.shape[0]:, 0:dims[1]] = im[-1, 0]
        new_im[0:dims[0], dims[1] + im.shape[1]:] = im[0, -1]
        new_im[dims[0] + im.shape[0]:, dims[1] + im.shape[1]:] = im[-1, -1]
        for i in range(dims[0]):
            new_im[i, dims[1]:dims[1] + im.shape[1]] = im[0, :]
            new_im[-i, dims[1]:dims[1] + im.shape[1]] = im[-1, :]
        for j in range(dims[1]):
            new_im[dims[0]:dims[0] + im.shape[0], j] = im[:, 0]
            new_im[dims[0]:dims[0] + im.shape[0], -j] = im[:, -1]
        return new_im
