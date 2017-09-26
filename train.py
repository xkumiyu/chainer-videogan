import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions

from datasets import VideoDataset
from net import Discriminator
from net import Generator
from updater import VideoGANUpdater
from visualize import out_generated_video


def _get_images_paths(dataset, root):
    path_list = []
    with open(dataset) as paths_file:
        for path in paths_file:
            path = os.path.join(root, path.strip())
            all_files = os.listdir(path)
            image_files = [os.path.join(path, f) for f in all_files if ('png' in f or 'jpg' in f)]
            path_list.append(image_files)
    return path_list


def main():
    parser = argparse.ArgumentParser(description='VideoGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of videos in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', required=True,
                        help='Directory list of images files')
    parser.add_argument('--root', default='.',
                        help='Path of images files')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--video_codecs', default='H264',
                        help='Video Codec')
    parser.add_argument('--video_ext', default='avi',
                        help='Extension of output video files')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))

    gen = Generator()
    dis = Discriminator()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    train = VideoDataset(paths=_get_images_paths(args.dataset, args.root))
    print('# data-size: {}'.format(len(train)))
    print('# data-shape: {}'.format(train[0].shape))
    print('')

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    updater = VideoGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={'gen': opt_gen, 'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_generated_video(
            gen, dis, 5, args.seed, args.out,
            args.video_codecs, args.video_ext),
        trigger=snapshot_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    if args.gpu >= 0:
        gen.to_cpu()
    chainer.serializers.save_npz(os.path.join(args.out, 'gen.npz'), gen)


if __name__ == '__main__':
    main()
