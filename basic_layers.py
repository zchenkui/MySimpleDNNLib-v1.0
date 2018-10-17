import numpy as np 
from basic_functions import softmax, cross_entropy_error


class MatMul:
    """ Matrix Multiplying Block

    MatMul is initialized by a weight matrix W (size: D * H) 
    and performs forward and backward operations.
    
    Given an input mini-batch x (size: N * D), MatMul outputs 
    the result of dot-product of x (N * D) and W (D * H). The 
    size of output is (N * H).

    Given a derivative matrix dout (size: N * H), MatMul performs 
    back-propagation to calculate the gradients of x and W, where 
    the sizes of dx and dW are (N * D) and (D * H) respectively. 
    """
    def __init__(self, W):
        """ Initialize MatMul class

        Parameter:
            W: a weight matrix with size (D * H)
        """
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
    
    def forward(self, x):
        """ Forward-propagation of MatMul

        Parameter:
            x: a mini-batch with size (N * D)
        
        Return:
            out: out (N * H) = dot(x (N * D), W (D * H))

        Save:
            x
        """
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out 

    def backward(self, dout):
        """ Back-propagation of MatMul

        Parameter:
            dout: a derivative matrix from above layer (size: N * H)

        Return:
            dx: dx (N * D) = dot(dout (N * H), W.T (H * D))
        
        Save:
            dW: dW (D * H) = dot(x.T (D * N), dout (N * H))
        """
        W, = self.params
        dx = np.dot(dout, np.transpose(W))
        dW = np.dot(np.transpose(self.x), dout)
        self.grads[0][...] = dW
        return dx


class Affine:
    """ Affine (or linear) Layer Block

    Affine is initialized by a weight matrix W (size: D * H) 
    and a bias vector b (size: H) and then performs forward 
    and backward operations.
    
    Given an input mini-batch x (size: N * D), Affine outputs 
    the result of x.W + b. The size of output is (N * H).

    Given a derivative matrix dout (size: N * H), Affine performs 
    back-propagation to calculate the gradients of x, W, and b, where 
    the sizes of dx, dW, and db are (N * D), (D * H), and H respectively. 
    """
    def __init__(self, W, b):
        """ Initialize Affine class

        Parameter:
            W: a weight matrix with size (D * H)
            b: a vector with size H
        """
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        """ Forward-propagation of Affine layer

        Parameter:
            x: a mini-batch with size (N * D)
        
        Return:
            out: out (N * H) = dot(x (N * D), W (D * H)) + b (H)

        Save:
            x
        """
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        """ Back-propagation of Affine layer

        Parameter:
            dout: a derivative matrix from above layer (size: N * H)

        Return:
            dx: dx (N * D) = dot(dout (N * H), W.T (H * D))
        
        Save:
            dW: dW (D * H) = dot(x.T (D * N), dout (N * H))
            db: db (H) = numpy.sum(dout (N * H), axis=0)
        """
        W, _ = self.params
        dx = np.dot(dout, np.transpose(W))
        dW = np.dot(np.transpose(self.x), dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Softmax:
    """ Softmax layer 

    There are no parameters and graduations in Softmax Layer.

    Softmax layer converts the input "score batch" to a related 
    "probability batch", in which each row indicates a probability 
    vector.

    A probability vector is a vector whose length is equal to the 
    number of classes. Each element in the vector is a probability 
    to a corresponding class and the highest probability denotes the 
    class predicted by the current model.
    """
    def __init__(self):
        """ Initialize Softmax class 

        Parameter:
            There is no parameter.
        """
        self.params = []
        self.grads = []
        self.out = None

    def forward(self, x):
        """ Forward-propagation of Softmax layer

        Parameter:
            x: input "score" matrix (size: N * D)
        
        Return:
            out: a probability matrix (size: N * D) given by softmax function

        Save:
            out (size: N * D)
        """
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        """ Back-propagation of Softmax layer

        Parameter:
            dout: a derivative matrix from above layer (size: N * D) 

        Return: 
            dx: dx (N * D) 
        """
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:
    """ SoftmaxWithLoss Layer

    SoftmaxWithLoss layer combines softmax function and cross entropy error function 
    so that the process will be easier. Note that there is no parameter in this class 
    either. 
    """
    def __init__(self):
        """ Initialize SoftmaxWithLoss class
        """
        self.params = []
        self.grads = []
        self.y = None # output of softmax layer
        self.t = None # the labels (true answer)

    def forward(self, x, t):
        """ Forward-propagation of SoftmaxWithLoss layer

        Parameters:
            x: "score" matrix with size (N * C) (N: batch size, C: the number of classes)
            t: "label" vector (N) or matrix (N * C)
                "label" vector (N): each element of t is a label of the corresponding row in the batch
                "label" matrix (N * C): each row of t is a one-hot vector with length C. Only the real 
                                        class is given 1 and the others are all 0.
        
        Return:
            loss: a scalar which indicates the loss of the mini-batch.
        """
        self.t = t
        self.y = softmax(x)

        if self.t.size == self.y.size:
            self.t = np.argmax(self.t, axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1): 
        """ Back-propagation of SoftmaxWithLoss layer

        Parameter:
            dout: a scalar initialized by user and often 1

        Return:
            dx: dx (N * C) = ((y (N * C) - t (N * C)) * dout (scalar)) / N
        """
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx /= batch_size
        return dx


class Sigmoid:
    """ Sigmoid layer

    Sigmoid layer is used to perform two labels classification.
    Given a mini-batch score matrix with size (N * 2), Sigmoid 
    layer converts the score matrix to a probability matrix (N * 2).

    Sigmoid layer has no parameter.
    """
    def __init__(self):
        """ Initialize Sigmoid class
        """
        self.params = []
        self.grads = [] 
        self.out = None 

    def forward(self, x):
        """ Forward-propagation of Sigmoid layer

        Parameter:
            x: a mini-batch score matrix (size: N * 2)

        Return:
            out: a corresponding probability matrix (size: N * 2)
        
        Save:
            out
        """
        out = 1. / (1. + np.exp(-x))
        self.out = out 
        return out 

    def backward(self, dout):
        """ Back-propagation of Sigmoid layer

        Parameter: 
            dout: a derivative matrix from above layer (size: N * 2)
        
        Return:
            dx: dx (N * 2) = dout (N * 2) * (1. - self.out (N * 2)) * self.out (N * 2)
        """
        dx = dout * (1. - self.out) * self.out
        return dx


class SigmoidWithLoss:
    """ Combine Sigmoid layer and cross entropy error

    SigmoidWithLoss is used to perform two classes classification.
    Given a score vector x and the corresponding labels t, SigmoidWithLoss 
    first converts x to a score vector and then calculates the loss of 
    the score vector x.

    This class has no parameters and gradients.
    """
    def __init__(self): 
        """ Initialize SigmoidWithLoss class

        No input parameter.
        """
        self.params = []
        self.grads = [] 
        self.loss = None 
        self.y = None
        self.t = None 

    def forward(self, x, t): 
        """ Forward propagation of SigmoidWithLoss class

        Parameters:
            x: a score vector with N elements, each of which identifies the score of the corresponding sample.
            t: a label vector with N elements, each of which is the label of the corresponding sample.

        Return:
            self.loss: the loss of x. (s scalar)
        """
        self.t = t
        self.y = 1. / (1. + np.exp(-x)) # convert the scores to probabilities
        self.loss = cross_entropy_error(np.c_[1-self.y, self.y], self.t) # calculate the loss
        return self.loss

    def backward(self, dout=1):
        """ Back propagation of SigmoidWithLoss class

        Parameters: 
            dout: a scalar that identifies the derivative of loss

        Return:
            dx: dx (N) = (y (N) - t (N)) * dout (scalar) / batch_size (scalar)
        """
        batch_size = np.shape(self.t)[0]
        dx = (self.y - self.t) * dout / batch_size
        return dx