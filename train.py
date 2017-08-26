import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions

from net import Discriminator
from net import Generator
from updater import VideoGANUpdater
from datasets import PreprocessedDataset


def main():
    parser = argparse.ArgumentParser(description='VideoGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of videos in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', required=True,
                        help='Directory of video files.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    # parser.add_argument('--seed', type=int, default=0,
    #                     help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--debug', '-d', action='store_true')
    args = parser.parse_args()

    if args.debug:
        chainer.config.debug = True
        print('Debug MODE')
        print()

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
        return optimizer
    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    all_files = os.listdir(args.dataset)
    video_files = [f for f in all_files if ('mp4' in f)]
    train = PreprocessedDataset(paths=video_files, root=args.dataset)
    print('# data-size: {}'.format(len(train)))
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
    # trainer.extend(
    #     out_generated_image(
    #         gen, dis,
    #         10, 10, args.seed, args.out),
    #     trigger=snapshot_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
