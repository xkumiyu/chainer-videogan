# Chainer-VideoGAN

Chainer implementation of [Generating Videos with Scene Dynamics](http://carlvondrick.com/tinyvideo/).

# Requirements

* Python 3.6
* FFmpeg
* OpenCV

# Install

```
pip install -r requirements/base.txt
```

# Usage

```
usage: train.py [-h] [--batchsize BATCHSIZE] [--epoch EPOCH] [--gpu GPU]
                --dataset DATASET [--out OUT] [--resume RESUME]
                [--snapshot_interval SNAPSHOT_INTERVAL]
                [--display_interval DISPLAY_INTERVAL] [--debug]

VideoGAN

optional arguments:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE, -b BATCHSIZE
                        Number of videos in each mini-batch
  --epoch EPOCH, -e EPOCH
                        Number of sweeps over the dataset to train
  --gpu GPU, -g GPU     GPU ID (negative value indicates CPU)
  --dataset DATASET, -i DATASET
                        Directory of video files.
  --out OUT, -o OUT     Directory to output the result
  --resume RESUME, -r RESUME
                        Resume the training from snapshot
  --snapshot_interval SNAPSHOT_INTERVAL
                        Interval of snapshot
  --display_interval DISPLAY_INTERVAL
                        Interval of displaying log to console
```
