import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


# TODO: 平均0, 標準偏差0.01のガウスノイズですべての重みを初期化
class Block2D(chainer.Chain):
    def __init__(self, out_channels, ksize=4, stride=2, pad=1):
        super(Block2D, self).__init__()
        with self.init_scope():
            self.dc = L.Deconvolution2D(None, out_channels, ksize, stride, pad)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.dc(x)
        h = self.bn(h)
        return F.relu(h)


class Block3D(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=4, stride=2, pad=1):
        super(Block3D, self).__init__()
        with self.init_scope():
            self.dc = L.DeconvolutionND(
                3, in_channels, out_channels, ksize, stride, pad)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.dc(x)
        h = self.bn(h)
        return F.relu(h)


class Generator(chainer.Chain):
    def __init__(self, noise_dim=100, bottom_time=2, bottom_width=4, ch=512):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.ch = ch
        self.bottom_time = bottom_time
        self.bottom_width = bottom_width
        # TODO: 第1層のksizeを2x4x4に

        with self.init_scope():
            self.bg_l = L.Linear(
                self.noise_dim, self.bottom_width * self.bottom_width * self.ch)
            self.bg_bn = L.BatchNormalization(
                self.bottom_width * self.bottom_width * self.ch)
            self.bg_block1 = Block2D(ch // 2)
            self.bg_block2 = Block2D(ch // 4)
            self.bg_block3 = Block2D(ch // 8)
            self.bg_dc = L.Deconvolution2D(
                ch // 8, 3, ksize=4, stride=2, pad=1)

            self.fg_l = L.Linear(
                self.noise_dim, self.bottom_time * self.bottom_width * self.bottom_width * self.ch)
            self.fg_bn = L.BatchNormalization(
                self.bottom_time * self.bottom_width * self.bottom_width * self.ch)
            self.fg_block1 = Block3D(ch, ch // 2)
            self.fg_block2 = Block3D(ch // 2, ch // 4)
            self.fg_block3 = Block3D(ch // 4, ch // 8)
            self.fg_dc = L.DeconvolutionND(
                3, ch // 8, 3, ksize=4, stride=2, pad=1)

            self.m_dc = L.DeconvolutionND(
                3, ch // 8, 1, ksize=4, stride=2, pad=1)

    def make_noize(self, batchsize):
        return np.random.randn(batchsize * self.noise_dim)\
            .reshape(batchsize, self.noise_dim).astype(np.float32)

    def __call__(self, z):
        batchsize = len(z)
        # Foreground
        fg_h = F.reshape(
            F.relu(self.fg_bn(self.fg_l(z))),
            (batchsize, self.ch, self.bottom_time, self.bottom_width, self.bottom_width))
        fg_h = self.fg_block1(fg_h)
        fg_h = self.fg_block2(fg_h)
        fg_h = self.fg_block3(fg_h)
        # TODO: シグモイド関数をかけたあとにスパースさせる
        # we also add to the objective a small sparsity prior on the mask λ||m(z)|| for λ = 0.1
        mask = F.sigmoid(self.m_dc(fg_h))
        fg = F.tanh(self.fg_dc(fg_h))

        # Background
        bg_h = F.reshape(F.relu(self.bg_bn(self.bg_l(z))),
                         (batchsize, self.ch, self.bottom_width, self.bottom_width))
        bg_h = self.bg_block1(bg_h)
        bg_h = self.bg_block2(bg_h)
        bg_h = self.bg_block3(bg_h)
        bg = F.tanh(self.bg_dc(bg_h))

        # Generated Video
        # TODO: ハードコーディングをやめる
        # mask shape: (batchsize, 1, 32, 64, 64) -> (batchsize, 3, 32, 64, 64)
        mask = F.tile(mask, (1, 3, 1, 1, 1))
        # bg shape: (batchsize, 3, 64, 64) -> (batchsize, 3, 32, 64, 64)
        bg = F.tile(F.reshape(bg, (batchsize, 3, 1, 64, 64)), (1, 1, 32, 1, 1))
        x = mask * fg + (1 - mask) * bg
        return x


class Discriminator(chainer.Chain):
    def __init__(self, bottom_time=2, bottom_width=4, ch=512):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.c0 = L.ConvolutionND(3, 3, ch // 8, ksize=4, stride=2, pad=1)
            self.c1 = L.ConvolutionND(3, ch // 8, ch // 4, ksize=4, stride=2, pad=1)
            self.c2 = L.ConvolutionND(3, ch // 4, ch // 2, ksize=4, stride=2, pad=1)
            self.c3 = L.ConvolutionND(3, ch // 2, ch, ksize=4, stride=2, pad=1)
            self.c4 = L.ConvolutionND(3, ch, ch, ksize=(2, 4, 4), stride=2, pad=0)
            self.l5 = L.Linear(ch, 1)
            self.bn1 = L.BatchNormalization(ch // 4)
            self.bn2 = L.BatchNormalization(ch // 2)
            self.bn3 = L.BatchNormalization(ch)
            self.bn4 = L.BatchNormalization(ch)

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        h = F.leaky_relu(self.bn4(self.c4(h)))
        return self.l5(h)
