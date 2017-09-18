import os

import numpy as np
import six
import cv2
import chainer


def _read_video_as_array(path, dtype):
    """
    return video shape: (frame, height, width, ch)
    """
    video = []
    cap = cv2.VideoCapture(path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    return np.asarray(video, dtype=dtype)


class VideoDataset(chainer.dataset.DatasetMixin):
    def __init__(self, paths, root='.', dtype=np.float32):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root = root
        self._dtype = dtype

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        """
        return video shape: (ch, frame, width, height)
        """
        path = os.path.join(self._root, self._paths[i])
        video = _read_video_as_array(path, self._dtype)

        if len(video.shape) != 4:
            raise ValueError('invalid video.shape')
        return video.transpose(3, 0, 2, 1)


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, paths, root, frame_size=32, image_size=(64, 64)):
        self.base = VideoDataset(paths, root)
        self.frame_size = frame_size
        self.image_size = image_size

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        video = self.base[i]

        if video.shape[1] < self.frame_size:
            raise Exception('Frame size of video[{}] is less than {}.'.format(i, self.frame_size))

        if video.shape[1] != self.frame_size:
            print('Warning: Frame size of video[{}] is not equal to {}.'.format(i, self.frame_size))

        video = video[:self.frame_size]
        video = 2 * video / 255. - 1

        return video
