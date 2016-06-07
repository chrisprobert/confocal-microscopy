import numpy as np
from scipy import ndimage



class ConvOp(object):
    '''Base class for a convolutional operation'''

    def __init__(self, name, shape=None):
        assert(np.sum(np.mod(np.array(shape), 2) - 1) == 0)
        self.name = name
        self.shape = shape

    def apply(self, target):
        raise NotImplementedError('Not implemented in base class')


class ElementWiseMultiplyConvOp(ConvOp):

    def __init__(self, weights, name='ElementWiseMultiplyConvOp'):
        super(ElementWiseMultiplyConvOp, self).__init__(name, weights.shape)
        self.w = weights

    def apply(self, target):
        return np.sum(self.w * target)


class UniformBlurConvOp(ElementWiseMultiplyConvOp):

    def __init__(self, blur_size=3, name='UniformBlurConvOp'):
        weight = np.ones((blur_size, blur_size))
        super(UniformBlurConvOp, self).__init__(weight, name)


class BlobDetector(ElementWiseMultiplyConvOp):

    def __init__(self, name='BlobDetector'):
        weights_size = self.get_filter_size()
        weights = self.get_dispersion(weights_size)
        super(BlobDetector, self).__init__(weights, name)

    @staticmethod
    def get_x_coord_mat(shape):
        return np.tile(np.abs(np.arange(shape[1]) - np.floor(shape[1] / 2)), (shape[1], 1))

    @staticmethod
    def get_y_coord_mat(shape):
        return np.tile(np.abs(np.arange(shape[0]).reshape(shape[0], 1) - np.floor(shape[0] / 2)), (1, shape[0]))

    @staticmethod
    def get_coord_distance_mat(shape):
        D1 = np.tile(
            np.abs(np.arange(shape[1]) - np.floor(shape[1] / 2)), (shape[1], 1))
        D2 = np.tile(np.abs(np.arange(shape[0]).reshape(
            shape[0], 1) - np.floor(shape[0] / 2)), (1, shape[0]))
        D = np.power(np.power(D1, 2.) + np.power(D2, 2.), 0.5)
        return D

    def get_dispersion(self, shape):
        raise NotImplementedError('Not implemented in base class')

    def get_filter_size(self):
        raise NotImplementedError('Not implemented in base class')


class LaplacianBlobDetector(BlobDetector):

    def __init__(self, spread, name='LaplacianBlobDetector'):
        self.spread = spread
        super(LaplacianBlobDetector, self).__init__(name)

    def get_dispersion(self, shape):
        D = self.get_coord_distance_mat(shape) ** 2.0
        scale2 = self.spread ** 2.0
        disp = np.power(np.pi * scale2, -1.)
        disp = disp * ((D / (2. * scale2)) - 1.)
        disp = disp * np.exp(-D / (2. * scale2))
        return disp

    def get_filter_size(self):
        fsize = int(self.spread * 6)
        if not (fsize & 0x1):
            fsize += 1
        return (fsize, fsize)


class ConvFilter(object):
    '''Base convolutional filter class'''

    def __init__(self, conv_op, name='ConvFilter', extender=PerpendicularEdgeExtender()):
        self.name = name
        self.c = conv_op
        self.e = extender

    def apply(self, im):
        im_e = self.e.apply(im, self.c.shape)
        res = np.zeros(im_e.shape)
        hw = np.floor((self.c.shape[0]) / 2).astype(int)
        hh = np.floor((self.c.shape[1]) / 2).astype(int)
        for i in range(self.c.shape[0], self.c.shape[0] + im.shape[0]):
            for j in range(self.c.shape[1], self.c.shape[1] + im.shape[1]):
                res[i, j] = self.c.apply(
                    im_e[i - hw:i + hw + 1, j - hh:j + hh + 1])
        return res[self.c.shape[0]:-self.c.shape[0], self.c.shape[1]:-self.c.shape[1]]


class IterativeConvFilter(ConvFilter):

    def __init__(self, conv_ops, name='IterativeConvFilter', extender=PerpendicularEdgeExtender()):
        super(IterativeConvFilter, self).__init__(None, name, extender)
        self.cs = conv_ops
        self.max_dims = (max(map(lambda c: c.shape[0], conv_ops)), max(
            map(lambda c: c.shape[1], conv_ops)))

    def apply(self, im):
        im_e = self.e.apply(im, self.max_dims)

        def apply_on_im_e(c):
            return self.apply_conv_op(im_e, c)
        return list(map(apply_on_im_e, self.cs))

    def apply_conv_op(self, im, c):
        res = np.zeros(im.shape)
        hw = np.floor((c.shape[0]) / 2).astype(int)
        hh = np.floor((c.shape[1]) / 2).astype(int)
        for i in range(c.shape[0], im.shape[0] - c.shape[0]):
            for j in range(c.shape[1], im.shape[1] - c.shape[1]):
                res[i, j] = c.apply(im[i - hw:i + hw + 1, j - hh:j + hh + 1])
        return res[self.max_dims[0]:-self.max_dims[0], self.max_dims[1]:-self.max_dims[1]]


class HarrisDetector(object):

    def __init__(self, sigma_i, sigma_d, alpha):
        self.sigma_i = sigma_i
        self.sigma_d = sigma_d
        self.alpha = alpha

    @staticmethod
    def get_grads_xy(X, mode='reflect', cval=0.):
        '''Compute gradients using 0-padding'''
        # N.B.: using ndimage for speed, was too slow with numpy gradient
        Gx = ndimage.sobel(X, axis=1, mode=mode, cval=cval)
        Gy = ndimage.sobel(X, axis=0, mode=mode, cval=cval)
        return Gx, Gy

    def get_M(self, X, mode='reflect', cval=0.):
        '''Get the Harris-Laplace scale-adapted second moment matrix'''
        # Compute the gaussian smoothed image
        # N.B.: using ndimage for speed; could replace with gaussian filter
        # class above
        G = ndimage.gaussian_filter(X, self.sigma_d, mode=mode, cval=cval)

        # Compute derivatives of gaussian-smoothed image in x and y directions
        Lx, Ly = self.get_grads_xy(G, mode=mode, cval=cval)

        # Compute second derivatives from first derivatives in x and y
        # directions
        Lxx, Lxy = self.get_grads_xy(Lx, mode=mode, cval=cval)
        _, Lyy = self.get_grads_xy(Ly, mode=mode, cval=cval)

        # Convolve each second derivative matrix with the integration gaussian
        # N.B.: using ndimage for speed; could replace with gaussian filter
        # class above
        Lxx = ndimage.gaussian_filter(Lxx, self.sigma_i, mode=mode, cval=cval)
        Lxy = ndimage.gaussian_filter(Lxy, self.sigma_i, mode=mode, cval=cval)
        Lyy = ndimage.gaussian_filter(Lyy, self.sigma_i, mode=mode, cval=cval)

        # Get an empty matrix and store the second derivatives
        M = np.empty((*X.shape, 2, 2))
        M[:, :, 0, 0] = Lxx
        M[:, :, 1, 0] = Lxy
        M[:, :, 0, 1] = Lxy
        M[:, :, 1, 1] = Lyy

        # Apply scale correction
        M = (self.sigma_d ** 2.) * M
        return M

    def get_cornerness_from_M(self, X):
        '''Compute the cornerness metric from a second moment matrix M'''
        assert(X.ndim == 4)
        det = (X[:, :, 0, 0] * X[:, :, 1, 1]) - (X[:, :, 0, 1] * X[:, :, 1, 0])
        tr = X[:, :, 0, 0] + X[:, :, 1, 1]
        cornerness = det - (self.alpha * (tr ** 2.))
        return cornerness

    def harris(self, X, mode='reflect', cval=0.):
        M = self.get_M(X, mode=mode, cval=cval)
        C = self.get_cornerness_from_M(M)
        return C
