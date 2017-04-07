#!/usr/bin/env python

import numpy as np
import chainer
from chainer import training, Variable
from chainer import datasets, iterators, optimizers
from chainer import Chain
from chainer.training import extensions

import chainer.functions as F
import chainer.links as L



class MyModel(chainer.Chain):
    # Define model to be called later by L.Classifer()

    def __init__(self, n_out):
        super(MyModel, self).__init__(
            l1=L.Linear(None, n_out),
        )
    
    def __call__(self, x):
        return self.l1(x)



def generate_data():
    #Need to reshape so that each input is an array.
    reshape = lambda x: np.reshape(x, (len(x),1))

    X = np.linspace(-1, 1, 101).astype(np.float32)
    Y = (2 * X + np.random.randn(*X.shape) * 0.33).astype(np.float32)
    return reshape(X),reshape(Y)
    

def main():
    epoch = 100
    batch_size = 1

    data = generate_data()
    
    # Convert to set of tuples (target, label).
    train = datasets.TupleDataset(*data)

    model = L.Classifier(MyModel(1), lossfun=F.mean_squared_error)

    # Set compute_accuracy=False when using MSE.
    model.compute_accuracy=False

    # Define optimizer (Adam, RMSProp, etc)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Define iterators.
    train_iter = chainer.iterators.SerialIterator(train, batch_size)

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (epoch, 'epoch'))
    
    # Helper functions (extensions) to monitor progress on stdout.
    report_params = [
        'epoch', 
        'main/loss', 
        ]
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(report_params))
    trainer.extend(extensions.ProgressBar())

    # Run trainer
    trainer.run()

    # Should print out value close to 2.
    print(model.predictor(np.array([[1]]).astype(np.float32)).data)

if __name__=="__main__":
    main()
