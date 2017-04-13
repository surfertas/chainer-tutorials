#!/usr/bin/env python
# Reference: http://deeplearning.net/tutorial/dA.html

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import argparse
import numpy as np
import cupy as cp
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import training, Variable
from chainer import datasets, iterators, optimizers
from chainer import Chain
from chainer.training import extensions
from chainer.datasets import tuple_dataset


def visualize_image(images, save_name):
    """Helper function for visualizing reconstructions.
    Params:
        images: array images to process.
        save_name: file name to save output to.
    """
    dim = images.shape[0]
    n_image_rows = int(np.ceil(np.sqrt(dim)))
    n_image_cols = int(np.ceil(dim * 1.0 / n_image_rows))
    gs = gridspec.GridSpec(n_image_rows, n_image_cols, top=1., bottom=0.,
                           right=1., left=0., hspace=0., wspace=0.)

    for g, count in zip(gs, range(int(dim))):
        ax = plt.subplot(g)
        ax.imshow(images[count, :].astype(np.float32).reshape((28, 28)))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(save_name + '_vis.png')


def get_corrupted_input(x, corrupt_lvl):
    """Returns the corrupted version of the input x.
    Params:
        x: Image array to corrupt.
        corrupt_lvl: Amount of corruption, the higher the value the more noise.
    Returns:
        image: Corrupted image array.
    """
    mask = np.random.RandomState(1).binomial(size=x.shape, n=1, p=(1. - corrupt_lvl))
    return mask.astype(np.float32) * x


class MyDAE(chainer.Chain):
    # Define denoising autoencoder to be called later by Classifier()

    def __init__(self, n_inputs, n_hidden):
        super(MyDAE, self).__init__(
            encoder=L.Linear(n_inputs, n_hidden)
        )
        self.add_param('decoder_bias', n_inputs)
        # Need to initialize 'decoder_bias' or will get a bunch of nans.
        self.decoder_bias.data[...] = 0.

    def __call__(self, x):
        # Can tie the weights by defining the decoder operation here using the
        # F.transpose function, and the 'decoder_bias' which we added above.
        # https://github.com/pfnet/chainer/issues/34
        h = F.sigmoid(self.encoder(x))
        h = F.linear(h, F.transpose(self.encoder.W), self.decoder_bias)
        return F.sigmoid(h)


def main():
    parser = argparse.ArgumentParser(description='Chainer-Tutorial: CNN')
    parser.add_argument('--batch_size', '-b', type=int, default=128,
                        help='Number of samples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of times to train on data set')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID: -1 indicates CPU')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    # Set corruption level at the command line, 0.3 is default.
    parser.add_argument('--corrupt_level', '-c', default=0.3,
                        help='Sets the corruption level')
    args = parser.parse_args()

    # Load mnist data
    # http://docs.chainer.org/en/latest/reference/datasets.html
    train, test = chainer.datasets.get_mnist(withlabel=False)
    corrupted = get_corrupted_input(train, args.corrupt_level)
    # Generate data set, with input being original training data and the target,
    # the corrupted image.
    data = test_tup = tuple_dataset.TupleDataset(train, corrupted)

    # Define iterators.
    train_iter = chainer.iterators.SerialIterator(data, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test_tup, args.batch_size,
                                                 repeat=False, shuffle=False)

    # Current bottleneck dimension set at 10. Worth changing around to visualize
    # the sensitivity to this parameter.
    model = L.Classifier(MyDAE(784, 10), lossfun=F.mean_squared_error)
    model.compute_accuracy = False

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Set up trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))

    # Evaluate the model at end of each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))

    # Helper functions (extensions) to monitor progress on stdout.
    report_params = [
        'epoch',
        'main/loss',
        'validation/main/loss',
        'elapsed_time'
    ]
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(report_params))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.LogReport())

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'validation/main/loss'],
                'epoch', file_name='loss.png'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run trainer
    trainer.run()

    # If running with GPU need to use cupy when predicting with forward pass or
    # will get type error warning.
    xp = cp if args.gpu >= 0 else np

    imgs = []
    for i in range(100):
        pred = model.predictor(xp.array([test[i]]).astype(np.float32))
        imgs.append(pred.data[0])

    visualize_image(np.array(imgs), "test")


if __name__ == "__main__":
    main()
