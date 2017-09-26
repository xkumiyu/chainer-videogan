import os
import unittest

import chainer
from chainer.dataset import convert
from chainer import Variable
import numpy as np

from datasets import PreprocessedDataset
from net import Discriminator
from net import Generator


class TestDetaset(unittest.TestCase):
    pass


class TestNet(unittest.TestCase):
    def setUp(self):
        self.gpu = -1
        self.batchsize = 1
        self.width = 64
        self.height = 64
        self.frame = 32
        self.dataset = 'data/preprocessed/'

    def test_discriminate_fakevideo(self):
        dis = Discriminator()
        gen = Generator()

        z = Variable(np.asarray(gen.make_noize(self.batchsize)))
        self.assertEqual((self.batchsize, 100), z.shape)

        x_fake = gen(z)
        self.assertEqual((self.batchsize, 3, self.frame, self.height, self.width), x_fake.shape)

        y_fake = dis(x_fake)
        self.assertEqual((self.batchsize, 1), y_fake.shape)

    def test_realvideo(self):
        dis = Discriminator()

        all_files = os.listdir(self.dataset)
        video_files = [f for f in all_files if ('mp4' in f)]

        train = PreprocessedDataset(paths=video_files, root=self.dataset)
        train_iter = chainer.iterators.SerialIterator(train, self.batchsize)
        batch = train_iter.next()

        x_real = Variable(convert.concat_examples(batch, self.gpu))
        self.assertEqual((self.batchsize, 3, self.frame, self.height, self.width), x_real.shape)

        y_real = dis(x_real)
        self.assertEqual((self.batchsize, 1), y_real.shape)


if __name__ == "__main__":
    unittest.main()
