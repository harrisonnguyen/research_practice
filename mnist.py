import os, struct
import struct
import numpy as np

def read(digits, dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.
    """
    if dataset is "training":
        fname_img = os.path.join(path, 'MNIST_data/train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'MNIST_data/train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 'MNIST_data/t10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'MNIST_data/t10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.int8)
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    images =  np.zeros((len(ind),rows,cols))
    labels = np.zeros((len(ind),1))
    for i in xrange(len(ind)):
        images[i, :] = img[ ind[i],:,:]
        labels[i] = lbl[ind[i]]

    return images, labels

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
