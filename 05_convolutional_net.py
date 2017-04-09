#!/usr/bin/env python

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import training, Variable
from chainer import datasets, iterators, optimizers
from chainer import Chain
from chainer.training import extensions



class MyCNN(chainer.Chain):
    # Define model to be called later by Classifer()
    # Convolutional neural network.
    def __init__(self, n_out):
        super(MyCNN, self).__init__(
            # Convolution2D(in_channels, out_channels, ksize, stride, pad)
            # Since kernal size is ksize=3, to keep the size of input the "SAME",
            # using tensorflow lingo, need to set pad=1. (ex. if ksize=5, then
            # would need to set pad=2.
            # ex. out_dim_width = ((in_dim_width - ksize + 2*pad)/stride) + 1
            conv1=L.Convolution2D(1, 32, ksize=3, stride=1, pad=1),
            conv2=L.Convolution2D(32, 64, ksize=3, stride=1, pad=1),
            conv3=L.Convolution2D(64, 128, ksize=3, stride=1, pad=1),
            # Note that chainer abstracts the reshape step here.
            fc4=L.Linear(2048, 625),
            fc5=L.Linear(625,n_out)
        )
    
    def __call__(self, x):
        # Introduce the use of 'max_pooling_2d'. Need to sensitive on definition
        # of ksize, stride, and pad to get appropriate reduction in dimensions.
        # The "-> (batch_size, width, height, output_dim)", is the resulting
        # shape after the 'max_pooling_2d' operation.
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2, pad=0)
        h = F.dropout(h, ratio=0.3, train=True) # -> (None, 14, 14, 32)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=2, stride=2, pad=0)
        h = F.dropout(h, ratio=0.3, train=True) # -> (None, 7, 7, 64)    
        h = F.max_pooling_2d(F.relu(self.conv3(h)), ksize=2, stride=2, pad=0)
        h = F.dropout(h, ratio=0.3, train=True) # -> (None, 4, 4, 128)
        h = F.dropout(F.relu(self.fc4(h)), ratio=0.3, train=True)
        return self.fc5(h)


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
    args = parser.parse_args()


    # Load mnist data
    # http://docs.chainer.org/en/latest/reference/datasets.html
    # Setting ndim=3, returns samples with shape (1, 28, 28)
    train, test = chainer.datasets.get_mnist(ndim=3)

    # Define iterators.   
    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size,
                                                 repeat=False, shuffle=False)

    # Initialize model: Loss function defaults to softmax_cross_entropy.
    model = L.Classifier(MyCNN(10))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.RMSprop(lr=0.001, alpha=0.9)
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
        'main/accuracy',    
        'validation/main/accuracy',
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
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run trainer
    trainer.run()


if __name__=="__main__":
    main()



"""
epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
...
90          0.0591103   0.0962811             0.985774       0.982298                  997.147       
91          0.0652343   0.0788602             0.984508       0.982397                  1008.21       
92          0.0668617   0.0942615             0.984859       0.974288                  1019.28       
93          0.0661004   0.087646              0.985175       0.981606                  1030.29       
94          0.0687926   0.0815414             0.983425       0.97854                   1041.31       
95          0.0685477   0.0722562             0.984108       0.981112                  1052.33       
96          0.0668571   0.0732176             0.984675       0.9821                    1063.32       
97          0.0725783   0.0779541             0.984442       0.98032                   1074.36       
98          0.0733218   0.0798261             0.984308       0.979727                  1085.38       
99          0.0739939   0.0813479             0.984258       0.977156                  1096.41       
100         0.0739011   0.0698339             0.984442       0.983584                  1107.4 
"""
