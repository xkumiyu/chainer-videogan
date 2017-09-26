import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


def add_noise(h, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape)
    else:
        return h


class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__()

        with self.init_scope():
            w = chainer.initializers.Normal(0.01)

            self.fg_dc0 = L.DeconvolutionND(3, 100, 512, (2, 4, 4), initialW=w)
            self.fg_dc1 = L.DeconvolutionND(3, 512, 256, 4, 2, 1, initialW=w)
            self.fg_dc2 = L.DeconvolutionND(3, 256, 128, 4, 2, 1, initialW=w)
            self.fg_dc3 = L.DeconvolutionND(3, 128, 64, 4, 2, 1, initialW=w)
            self.fg_dc4 = L.DeconvolutionND(3, 64, 3, 4, 2, 1, initialW=w)
            self.m_dc4 = L.DeconvolutionND(3, 64, 1, 4, 2, 1, initialW=w)
            self.fg_bn0 = L.BatchNormalization(512)
            self.fg_bn1 = L.BatchNormalization(256)
            self.fg_bn2 = L.BatchNormalization(128)
            self.fg_bn3 = L.BatchNormalization(64)

            self.bg_dc0 = L.Deconvolution2D(100, 512, 4, initialW=w)
            self.bg_dc1 = L.Deconvolution2D(512, 256, 4, 2, 1, initialW=w)
            self.bg_dc2 = L.Deconvolution2D(256, 128, 4, 2, 1, initialW=w)
            self.bg_dc3 = L.Deconvolution2D(128, 64, 4, 2, 1, initialW=w)
            self.bg_dc4 = L.Deconvolution2D(64, 3, 4, 2, 1, initialW=w)
            self.bg_bn0 = L.BatchNormalization(512)
            self.bg_bn1 = L.BatchNormalization(256)
            self.bg_bn2 = L.BatchNormalization(128)
            self.bg_bn3 = L.BatchNormalization(64)

    def make_noize(self, batchsize):
        return np.random.randn(batchsize * 100)\
            .reshape(batchsize, 100).astype(np.float32)

    def foreground(self, z):
        """Foreground

        input z shape:     (batchsize, 100)
        output fg shape:   (batchsize, 3, 32, 64, 64)
        output mask shape: (batchsize, 1, 32, 64, 64)

        """
        batchsize = len(z)
        z = z.reshape(batchsize, 100, 1, 1, 1)
        h = F.relu(self.fg_bn0(self.fg_dc0(z)))
        h = F.relu(self.fg_bn1(self.fg_dc1(h)))
        h = F.relu(self.fg_bn2(self.fg_dc2(h)))
        h = F.relu(self.fg_bn3(self.fg_dc3(h)))
        mask = F.dropout(F.sigmoid(self.m_dc4(h)), ratio=0.5)
        fg = F.tanh(self.fg_dc4(h))
        return fg, mask

    def background(self, z):
        """Background

        input z shape:  (batchsize, 100)
        output h shape: (batchsize, 3, 64, 64)

        """
        batchsize = len(z)
        z = z.reshape(batchsize, 100, 1, 1)
        h = F.relu(self.bg_bn0(self.bg_dc0(z)))
        h = F.relu(self.bg_bn1(self.bg_dc1(h)))
        h = F.relu(self.bg_bn2(self.bg_dc2(h)))
        h = F.relu(self.bg_bn3(self.bg_dc3(h)))
        return F.tanh(self.bg_dc4(h))

    def __call__(self, z):
        """Call

        input z shape:  (batchsize, 100) or (batchsize, 1024, 4, 4)
        output x shape: (batchsize, 3, 32, 64, 64)

        """
        # TODO(@xkumiyu) Correspond to (batchsize, 1024, 4, 4)
        batchsize = len(z)

        fg, mask = self.foreground(z)
        mask = F.tile(mask, (1, 3, 1, 1, 1))

        bg = self.background(z)
        bg = F.reshape(bg, (batchsize, 3, 1, 64, 64))
        bg = F.tile(bg, (1, 1, 32, 1, 1))

        x = mask * fg + (1 - mask) * bg
        return x


class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__()

        with self.init_scope():
            w = chainer.initializers.Normal(0.01)
            self.l5 = L.Linear(512, 1, initialW=w)
            self.conv0 = L.ConvolutionND(3, 3, 64, 4, 2, 1, initialW=w)
            self.conv1 = L.ConvolutionND(3, 64, 128, 4, 2, 1, initialW=w)
            self.conv2 = L.ConvolutionND(3, 128, 256, 4, 2, 1, initialW=w)
            self.conv3 = L.ConvolutionND(3, 256, 512, 4, 2, 1, initialW=w)
            self.conv4 = L.ConvolutionND(3, 512, 512, (2, 4, 4), initialW=w)
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(256)
            self.bn3 = L.BatchNormalization(512)
            self.bn4 = L.BatchNormalization(512)

    def __call__(self, x):
        """Call

        input x shape:  (batchsize, 3, 32, 64, 64)
        output y shape: (batchsize, 1)

        """
        h = F.leaky_relu(self.conv0(add_noise(x)))
        h = F.leaky_relu(self.bn1(self.conv1(h)))
        h = F.leaky_relu(self.bn2(self.conv2(h)))
        h = F.leaky_relu(self.bn3(self.conv3(h)))
        h = F.leaky_relu(self.bn4(self.conv4(h)))
        return self.l5(h)
