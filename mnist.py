# python 3
import gzip, os
from struct import unpack
import numpy as np
from array import array


def load_mnist(dataset='train', path='.'):
    if dataset == "train":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte.gz')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte.gz')
    elif dataset == "test":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte.gz')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte.gz')
    else:
        raise ValueError("dataset must be 'test' or 'train'")

    with gzip.open(fname_img, 'rb') as f:
        mn, ni, nr, nc = unpack('>IIII', f.read(16))
        img_data = array('B', f.read())
        img_data = np.array(img_data, dtype=np.uint8)
        img_data = img_data.reshape(ni, nr, nc)

    with gzip.open(fname_lbl, 'rb') as f:
        mn, nl = unpack('>II', f.read(8))
        lbl_data = array('B', f.read())
        lbl_data = np.array(lbl_data, dtype=np.uint8)

    return img_data, lbl_data


if __name__ == '__main__':
    Xtr, Ytr = load_mnist('train', 'data/')
