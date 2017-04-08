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
    # 784 is dimension of the inputs, 625 is n_units in hidden layer
    # and 10 is the output dimension.
    model = L.Classifier(MLP(625, 10))

    # Set up GPU usage if necessary. args.gpu is a condition as well as an
    # identification when passed to get_device().
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


"""
# Expected output with no gpu.
epoch       main/loss   validation/main/loss  main/accuracy validation/main/accuracy  elapsed_time
...
87          0.287069    0.278754              0.917761       0.920293                 630.602       
88          0.286355    0.278692              0.918253       0.920985                 637.872       
89          0.28583     0.279103              0.91836        0.920194                 645.144       
90          0.285477    0.278245              0.918194       0.920589                 652.43        
91          0.284905    0.277772              0.91861        0.920886                 659.696       
92          0.284681    0.276569              0.918586       0.920688                 667.01        
93          0.283991    0.276406              0.918893       0.920985                 674.273       
94          0.283576    0.276226              0.918943       0.921677                 681.529       
95          0.283074    0.275656              0.91916        0.921776                 688.781       
96          0.282758    0.275749              0.919154       0.920688                 696.205       
97          0.282119    0.274949              0.91956        0.921084                 703.496       
98          0.28187     0.274234              0.919343       0.922468                 710.768       
99          0.281488    0.274292              0.91946        0.922073                 718.026       
100         0.280849    0.274322              0.919571       0.920589                 725.273
"""
