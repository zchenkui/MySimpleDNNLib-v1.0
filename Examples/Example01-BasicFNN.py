import numpy as np 
import matplotlib.pyplot as plt 
import sys, os
sys.path.append(os.pardir)
from basic_layers import Affine, Sigmoid, SoftmaxWithLoss
from read_dataset import Spiral
from grad_optimizers import SGD
from trainer import Trainer


class TwoLayerNet:
    """ Build a two layer neural network and perform predict and classification
    """
    def __init__(self, input_size, hidden_size, output_size):
        """ Initialize a two layer net model

        Parameter:
            input_size: the number of features of a single data
            hidden_size: the number of nodes of hidden layer
            output_size: the number of classes

        Structure:
            Affine Layer 1 --> Sigmoid Activation --> Affine Layer 2 --> SoftmaxWithLoss
        """
        I, H, O = input_size, hidden_size, output_size

        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        self.layers = [
            Affine(W1, b1), 
            Sigmoid(), 
            Affine(W2, b2), 
        ]
        self.loss_layer = SoftmaxWithLoss()

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        """ Given an input data, predict its class

        Parameter:
            x: input data
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        """ 
        Given a dataset (or mini-batch) and a corresponding label set, 
        calculate the loss of current model

        Parameters:
            x: a dataset or a mini-batch
            t: corresponding label set

        Return:
            loss: the loss of current model
        """
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        """ Training model with back-propagation

        Parameter:
            dout: a scalar set by user. It denotes the derivative of loss.

        Return:
            dout: note that the shape of output and input is different.
        """
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


# hyper-parameters
max_epoch = 300 
batch_size = 30 
hidden_size = 10 
learning_rate = 1.0

# load data and initialize model
spiral = Spiral()
x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(learning_rate)

# train model and plot result
trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch=max_epoch, batch_size=batch_size, eval_interval=10)
trainer.plot() 

# show classification results
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()