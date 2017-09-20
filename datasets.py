import os

import numpy as np
from PIL import Image
import chainer


def _read_images_as_array(paths, root, dtype):
    """
    return video shape: (frame, height, width, ch)
    """
    video = []
    for path in paths:
        path = os.path.join(root, path)
        f = Image.open(path)
        try:
            image = np.asarray(f, dtype=dtype)
            video.append(image)
        finally:
            if hasattr(f, 'close'):
                f.close()

    return np.asarray(video, dtype=dtype)


class VideoDataset(chainer.dataset.DatasetMixin):
    """
    paths: 2-dim array
    """
    def __init__(self, paths, root='.', dtype=np.float32):
        self._paths = paths
        self._root = root
        self._dtype = dtype

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        """
        return video shape: (ch, frame, width, height)
        """
        video = _read_images_as_array(self._paths[i], self._root, self._dtype)

        if len(video.shape) != 4:
            raise ValueError('invalid video.shape')

        return video.transpose(3, 0, 2, 1)
