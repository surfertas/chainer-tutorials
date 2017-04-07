#!/usr/bin/env python

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import training, Variable
from chainer import datasets, iterators, optimizers
from chainer import Chain
from chainer.training import extensions



class MyModel(chainer.Chain):
    # Define model to be called later by L.Classifer()

    def __init__(self, n_out):
        super(MyModel, self).__init__(
            l1=L.Linear(None, n_out),
        )
    
    def __call__(self, x):
        return self.l1(x)


def main():
    epoch = 100
    batch_size = 128


    # Load mnist data
    # http://docs.chainer.org/en/latest/reference/datasets.html
    train, test = chainer.datasets.get_mnist()


    # Define iterators.   
    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    test_iter = chainer.iterators.SerialIterator(test, batch_size,
                                                 repeat=False, shuffle=False)

    
    # Initialize model: Loss function defaults to softmax_cross_entropy.
    # Can keep same model used in linear regression.
    model = L.Classifier(MyModel(10))


    # Define optimizer (SGD, Adam, RMSProp, etc)
    # http://docs.chainer.org/en/latest/reference/optimizers.html
    optimizer = chainer.optimizers.SGD()
    optimizer.setup(model)


    # Set up trainer
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (epoch, 'epoch'))

    
    # Evaluate the model at end of each epoch
    trainer.extend(extensions.Evaluator(test_iter, model))


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
