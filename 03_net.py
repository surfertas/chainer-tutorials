#!/usr/bin/env python

import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import training, Variable
from chainer import datasets, iterators, optimizers
from chainer import Chain
from chainer.training import extensions



class MLP(chainer.Chain):
    # Define model to be called later by L.Classifer()
    # Basic MLP
    def __init__(self, n_unit, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(None, n_unit),
            l2=L.Linear(None, n_out)
        )
    
    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        return self.l2(h)


def main():
    # Introduce argparse for clarity and organization.
    # Starting to use higher capacity models, thus set up for GPU.
    parser = argparse.ArgumentParser(description='Chainer-Tutorial: MLP')
    parser.add_argument('--batch_size', '-b', type=int, default=128,
                        help='Number of samples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of times to train on data set')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID: -1 indicates CPU')
    args = parser.parse_args()


    # Load mnist data
    # http://docs.chainer.org/en/latest/reference/datasets.html
    train, test = chainer.datasets.get_mnist()


    # Define iterators.   
    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size,
                                                 repeat=False, shuffle=False)

    
    # Initialize model: Loss function defaults to softmax_cross_entropy.
    # 784 is dimension of the inputs, and 10 is the output dimension.
    model = L.Classifier(MLP(784, 10))

    # Set up GPU usage if necessary.
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Define optimizer (SGD, Adam, RMSProp, etc)
    # http://docs.chainer.org/en/latest/reference/optimizers.html
    optimizer = chainer.optimizers.SGD()
    optimizer.setup(model)


    # Set up trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))

    
    # Evaluate the model at end of each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))


    # Helper functions (extensions) to monitor progress on stdout.
    report_params = [
        'epoch', 
        'main/loss',
        'validation/main/loss',
        'main/accuracy',
        'validation/main/accuracy',
        'elapsed_time'
        ]
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(report_params))
    trainer.extend(extensions.ProgressBar())


    # Run trainer
    trainer.run()


if __name__=="__main__":
    main()
