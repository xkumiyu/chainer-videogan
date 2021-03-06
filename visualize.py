import os

import chainer
import chainer.cuda
from chainer import Variable
import cv2
import numpy as np


def _write_video(x, filepath, codecs, fps=25.0):
    frames, height, width, ch = x.shape

    fourcc = cv2.VideoWriter_fourcc(*codecs)
    video = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    for i in range(frames):
        video.write(x[i])
    video.release()


def out_generated_video(gen, dis, n_videos, seed, dst, codecs, ext):
    @chainer.training.make_extension()
    def make_video(trainer):
        np.random.seed(seed)
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_noize(n_videos)))
        with chainer.using_config('train', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        x = x.transpose(0, 2, 3, 4, 1)

        preview_dir = os.path.join(dst, 'preview', 'iter_{}'.format(trainer.updater.iteration))
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        for i in range(x.shape[0]):
            preview_path = os.path.join(preview_dir, '{}.{}'.format(i, ext))
            _write_video(x[i], preview_path, codecs)
    return make_video
