# Chainer-VideoGAN

Chainer implementation of [Generating Videos with Scene Dynamics](http://carlvondrick.com/tinyvideo/).

# Requirements

* Python 3.6
* FFmpeg
* OpenCV

# Installation

```
pip install -r requirements.txt
```

# Usage

## Preprocess

```
python preprocess.py <input data directory> <output data directory>
```

## Train

```
python train.py -i <input data list file>
```
