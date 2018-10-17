import numpy as np 
import sys, os
from word2vector import Embedding
from basic_functions import softmax, sigmoid
from basic_layers import Softmax


class RNN:
    """ Recurrent Neural Network model (single block)

    A single block of RNN, which is the base of time series nerual network.
    """
    def __init__(self, Wx, Wh, b):
        """ Initialize RNN 

        Parameters:
            Wx: affine matrix for x(t)
            Wh: affine matrix for h(t-1)
            b: bias vector
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None 

    def forward(self, x, h_prev):
        """ Forward propagation of RNN

        Parameters:
            x: x(t) is the tth item in time series. The size of x(t) is (1 * D) (a vector) or (N * D) (a mini-batch)
            h_pre: h(t-1) is the previous output of RNN. The size of h(t-1) is (1 * H) (a vector) or (N * H) (a mini-batch)
            Note that N is the batch size.

        Size of weights matrices:
            Wx: (D * H)
            Wh: (H * H)
            b: (1 * H)

        Return:
            h_next: h(t) is the output of the current RNN block, which will be sent to the next RNN block.
            the size of h_next: (1 * H) (a vector) or (N * H) (a mini-batch) 
            how to calculate h_next: h(t) = tanh(h(t-1).Wh + x(t).Wx + b)
        """
        Wx, Wh, b = self.params 
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b 
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        """ Back propagation of RNN

        Parameter
            dh_next: the derivative matrix of h_next (i.e. h(t)). The size of dh_next is (N * H), which is identical to h_next.

        The size of matrices in each step:
            dt: (N * H)
            db: (1 * H)
            dWh: (H * H)
            dh_prev: (N * H)
            dWx: (D * H)
            dx: (N * D)
            
        Procedure:
            step 1. dt = dh_next * (1 - h_next^2). Note that [tanh(x)]' = 1 - [tanh(x)]^2
            step 2. db = np.sum(dt, axis=0). Add all rows togethor, dt (N * H) ---> db (1 * H)
            step 3. dWh = transpose(h_prev).dt. h_prev (N * H) ---> transpose(h_prev) (H * N) ---> (H * N).(N * H) ---> (H * H)
            step 4. dh_prev = dt.transpose(Wh). transpose(Wh) (H * H) ---> dt (N * H) ---> (N * H).(H * H) ---> (N * H)
            step 5. dWx = transpose(x).dt. x (N * D) ---> transpose(x) (D * N) ---> dt (N * H) ---> (D * N).(N * H) ---> (D * H)
            step 6. dx = dt.transpose(Wx). dt (N * H) ---> transpose(Wx) (H * D) ---> (N * H).(H * D) ---> (N * D)

        """
        Wx, Wh, _ = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next**2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(np.transpose(h_prev), dt)
        dh_prev = np.dot(dt, np.transpose(Wh))
        dWx = np.dot(np.transpose(x), dt)
        dx = np.dot(dt, np.transpose(Wx))

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeRNN:
    """ Time RNN combines RNN blocks and learns time series

    """
    def __init__(self, Wx, Wh, b, stateful=False):
        """ Initialize TimeRNN

        Parameters:
            Wx: (D * H) matrix
            Wh: (H * H) matrix
            b: (1 * H) vector
            stateful: use h_prev or not (if not, set h_prev to 0)
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)] 
        self.layers = None 
        self.h, self.dh = None, None 
        self.stateful = stateful

    def set_state(self, h):
        """ Set hidden state to h

        Paramter:
            h: a hidden state matrix (size: (N * H))

        Return:
            None
        """
        self.h = h

    def reset_state(self):
        """ Reset hidden state to None
        """
        self.h = None

    def forward(self, xs):
        """ Forward propagation of TimeRNN

        Parameter:
            xs: a time series input. The size of xs is (N * T * D) where N is the batch size, T is the time, and D is the length of word vector
        
        Return:
            hs: a time series hidden state. The size of hs is (N * T * H) where H is the length of hidden vector

        Note:
            if stateful is False or h is None, the input h is set to an (N * H) zero matrix.
        """
        Wx, _, _ = self.params
        N, T, _ = np.shape(xs)
        _, H = np.shape(Wx)

        self.layers = []
        hs = np.empty((N, T, H), dtype="f")

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs 

    def backward(self, dhs):
        """ Back propagation of TimeRNN

        Parameter:
            dhs: the derivative matrix of hs. dhs has the same size with hs (N * T * H)

        Return:
            dxs: the derivative matrix of xs. dxs has the same size with xs (N * T * D)

        Note that the initial dh is always 0.
        """
        Wx, _, _ = self.params
        N, T, _ = np.shape(dhs)
        D, _ = np.shape(Wx)

        dxs = np.empty((N, T, D), dtype="f") 
        dh = 0 
        grads = [0, 0, 0] 
        for t in reversed(range(T)): 
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad 
        self.dh = dh

        return dxs


class TimeEmbedding: 
    """ Extend Embedding layer to time series

    """
    def __init__(self, W):
        """ Initialize TimeEmbedding class

        Parameter:
            W: an embedding matrix with size (V * D) where V is the vocabulary size
        """
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None 
        self.W = W 

    def forward(self, xs): 
        """ Forward propagation of TimeEmbedding

        Parameter:
            xs: a time series batch with size (N * T) and each element is a word id. If N = 2 and T = 5, 
                xs may look like:
                [
                    batch 0: [0, 1, 4, 8, 35], 
                    batch 1: [3, 6, 15, 7, 6],
                ]

        Return:
            out: convert each element in xs to a word vector with fixed length D. The size of out is (N * T * D)
        """
        N, T = np.shape(xs) 
        _, D = np.shape(self.W)

        out = np.empty((N, T, D))
        self.layers = []

        for t in range(T): 
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout): 
        """ Back propagation of TimeEmbedding

        Paramter:
            dout: the derivative matrix of out (see forward function). The size of dout is (N * T * D)

        Return:
            None
        
        Note that we add all grads togethor
        """
        T = np.shape(dout)[1]
        grad = 0

        for t in range(T): 
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad


class TimeAffine:
    """ Extend Affine layer to time series
    """
    def __init__(self, W, b): 
        """ Initialize TimeAffine class

        Paramters:
            W: an affine matrix with size (H * V) where H is the length of hidden vector and V is the vocabulary size
            b: a bias vector with V elements.
        """
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None 

    def forward(self, x):
        """ Forward propagation of TimeAffine

        Parameter:
            x: in RNN, x is identical to hs (see TimeRNN class). The size of x is (N * T * H)

        Return:
            out: a time series "score" matrix with size (N * T * V)
        """
        N, T, _ = np.shape(x)
        W, b = self.params

        rx = np.reshape(x, newshape=(N * T, -1)) 
        out = np.dot(rx, W) + b 
        out = np.reshape(out, newshape=(N, T, -1))
        self.x = x 
        return out 

    def backward(self, dout): 
        """ Backward propagation of TimeAffine

        Parameter:
            dout: the derivative matrix of out (i.e. the score matrix, see forward function) with size (N * T * V)

        Return:
            dx: the derivative matrix of x. Here dx and x are dhs and hs respectively (see TimeRNN class). The size of x (hs) and dx (dhs) is (N * T * H)
        """
        x = self.x 
        N, T, H = np.shape(x) 
        W, _ = self.params

        dout = np.reshape(dout, newshape=(N * T, -1))
        rx = np.reshape(x, newshape=(N * T, -1))

        db = np.sum(dout, axis=0)
        dW = np.dot(np.transpose(rx), dout)
        dx = np.dot(dout, np.transpose(W))
        dx = np.reshape(dx, newshape=(N, T, H))

        self.grads[0][...] = dW 
        self.grads[1][...] = db

        return dx 


class TimeSoftmaxWithLoss:
    """ TimeSoftmaxWithLoss class

    We combine softmax layer and loss layer togethor and then convert them to time series version
    """
    def __init__(self): 
        """ Initialize TimeSoftmaxWithLoss class

        There is no input argument

        self.ingore_label is used in dropout algorithm
        """
        self.params, self.grads = [], []
        self.cache = None 
        self.ignore_label = -1

    def forward(self, xs, ts): 
        """ Forward propagation of TimeSoftmaxWithLoss

        Parameters:
            xs: a score matrix from TimeAffine layer. The size of xs is (N * T * V), where V is the vocabulary size.
            ts: a label matrix. ts may have two various sizes:
                1. (N * T * V), each label is given as a one-hot vector with length V
                2. (N * T), each label is given as the target word id (a scalar)

        Return:
            loss: the loss of xs (a scalar)

        Note:
            how to calculate the loss of time series.

                    batch N-1             batch N-2        ......     batch 0
        t = 0   score_vec(N-1, 0)     score_vec(N-2, 0)    ...... score_vec(0, 0)    ---> ts(0, 0)  ---> Softmax and Cross Entropy Layer ---> loss 0    ---|
        t = 1   score_vec(N-1, 1)     score_vec(N-2, 1)    ...... score_vec(0, 1)    ---> ts(0, 1)  ---> Softmax and Cross Entropy Layer ---> loss 1    ---|
        .              .                     .                          .                    .                 .                                           | 
        .              .                     .                          .                    .                 .                                           | ---> Average ---> loss 
        .              .                     .                          .                    .                 .                                           |
        .              .                     .                          .                    .                 .                                           |
        t = T-1 score_vec(N-1, T-1)   score_vec(N-2, T-1)  ...... score_vec(0, T-1)  ---> ts(0, T-1) ---> Softmax and Cross Entropy Layer ---> loss T-1 ---|
        
        Size of matrices:
            xs: (N * T * V) ---> ((N * T) * V) (3D to 2D)
            ts: (N * T * V) ---> (N * T) ---> (1 * (N * T)) (3D (argmax)---> 2D (reshape) ---> 1D vector)
            ys: ((N * T) * V) (2D)
            ls: (1 * (N * T)) (1D vector)
            loss: scalar
        """
        N, T, V = np.shape(xs)

        if np.ndim(ts) == 3:
            ts = np.argmax(ts, axis=2)

        mask = (ts != self.ignore_label) 

        xs = np.reshape(xs, newshape=(N * T, V))
        ts = np.reshape(ts, newshape=(N * T))
        mask = np.reshape(mask, newshape=(N * T))

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask 
        loss = -1 * np.sum(ls)
        loss /= np.sum(mask)

        self.cache = (ts, ys, mask, (N, T, V))

        return loss
        
    def backward(self, dout=1):
        """ Back propagation of TimeSoftmaxWithLoss

        Parameter:
            dout: the derivative of loss (a scalar)

        Return:
            dx: the derivative matrix of xs (see forward function). Size of dx is (N * T * V)
        """
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= np.sum(mask)
        dx *= mask[:, np.newaxis]
        dx = np.reshape(dx, newshape=(N, T, V))

        return dx


class LSTM:
    """ LSTM single block

    """
    def __init__(self, Wx, Wh, b):
        """ Initialize LSTM class

        Parameters:
            Wx: (D * 4H) matrix
            Wh: (H * 4H) matrix
            b: a vector with length 4H
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None 

    def forward(self, x, h_prev, c_prev):
        """Forward propagation of LSTM

        Parameters:
            x: (N * D) matrix
            h_prev: (N * H) matrix
            c_prev: (N * H) matrix

        Return:
            h_next: (N * H) matrix
            c_next: (N * H) matrix

        Steps of forward propagation:
            Sizes of matrices:
                x: (N * D)
                Wx: (D * 4H)
                h_prev: (N * H)
                Wh: (H * 4H)
                b: a vector with length 4H
                f, g, i, o: (N * H)
                c_next, h_next: (N * H)

            Step 1. A = x.Wx + h_prev.Wh + b
            Step 2. f = A[:, 0:H], g = A[:, H:2H], i = A[:, 2H:3H], o = A[:, 3H:]
            step 3. f = sigmoid(f) (forget gate)
                    g = tanh(g) 
                    i = sigmoid(i) (input gate)
                    o = sigmoid(o) (output gate)
            step 4. c_next = c_prev * f + g * i
                    h_next = o * tanh(c_next)

        """
        Wx, Wh, b = self.params 
        _, H = np.shape(h_prev) 

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b 
        f = A[:, 0 : H]
        g = A[:, H : 2 * H]
        i = A[:, 2 * H : 3 * H] 
        o = A[:, 3 * H : ]

        f = sigmoid(f)
        g = np.tanh(g) 
        i = sigmoid(i) 
        o = sigmoid(o)

        c_next = c_prev * f + g * i 
        h_next = np.tanh(c_next) * o 

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next): 
        """ Back propagation of LSTM

        Parameters:
            dh_next: the derivative matrix of h_next (size: (N * H))
            dc_next: the derivative matrix of c_next (size: (N * H))

        Return:
            dx: the derivative matrix of x (size: (N * D))
            dh_prev: the derivative matrix of h_prev (size: (N * H))
            dc_prev: the derivative matrix of c_prev (size: (N * H))
        """
        Wx, Wh, _ = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        ds = dc_next + dh_next * o * (1 - np.tanh(c_next)**2)
        dc_prev = ds * f

        do = dh_next * np.tanh(c_next) 
        di = ds * g 
        dg = ds * i 
        df = c_prev * ds

        do *= o * (1 - o)
        di *= i * (1 - i) 
        dg *= (1 - g**2) 
        df *= f * (1 - f) 

        dA = np.hstack((df, dg, di, do)) 

        db = np.sum(dA, axis=0) 
        dWh = np.dot(np.transpose(h_prev), dA)
        dh_prev = np.dot(dA, np.transpose(Wh))
        dWx = np.dot(np.transpose(x), dA) 
        dx = np.dot(dA, np.transpose(Wx)) 

        self.grads[0][...] = dWx 
        self.grads[1][...] = dWh
        self.grads[2][...] = db 

        return dx, dh_prev, dc_prev


class TimeLSTM: 
    """ Apply LSTM to time series data

    """
    def __init__(self, Wx, Wh, b, stateful=False):
        """ Initialize TimeLSTM class

        Parameters:
            Wx: (D * 4H) matrix
            Wh: (H * 4H) matrix
            b: a vector with length 4H
            stateful: set False so that c and h are initialize with 0

            (see forward function of LSTM to get for information)
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)] 
        self.layers = None 

        self.h, self.c = None, None 
        self.dh = None 
        self.stateful = stateful

    def forward(self, xs): 
        """ Forward propagation of TimeLSTM

        Parameter:
            xs: (N * T * D) matrix

        Return:
            hs: (N * T * H) matrix
        """
        Wx, Wh, b = self.params 
        N, T, _ = np.shape(xs) 
        H = np.shape(Wh)[0] 

        self.layers = []
        hs = np.empty(shape=(N, T, H), dtype="f")

        if (not self.stateful) or (self.h is None): 
            self.h = np.zeros(shape=(N, H), dtype="f")
        if (not self.stateful) or (self.c is None): 
            self.c = np.zeros(shape=(N, H), dtype="f")

        for t in range(T): 
            layer = LSTM(Wx, Wh, b) 
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c) 
            hs[:, t, :] = self.h 

            self.layers.append(layer) 
        
        return hs 

    def backward(self, dhs): 
        """ Back propagation of TimeLSTM

        Parameter:
            dhs: the derivative matrix of hs (size: (N * T * H))

        Return:
            dxs: the derivative matrix of xs (size: (N * T * D))
        """
        Wx, _, _ = self.params
        N, T, _ = np.shape(dhs) 
        D = np.shape(Wx)[0]

        dxs = np.empty(shape=(N, T, D), dtype="f")
        dh, dc = 0, 0 

        grads = [0, 0, 0]
        for t in reversed(range(T)): 
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dh + dhs[:, t, :], dc) 
            dxs[:, t, :] = dx 
            for i, grad in enumerate(layer.grads): 
                grads[i] += grad

        for i, grad in enumerate(grads): 
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c 

    def reset_state(self): 
        self.h, self.c = None, None  


class TimeDropout:
    """ Dropout layer used in time series
    """
    def __init__(self, dropout_rate=0.5):
        """ Initialize TimeDropout class

        Parameter:
            dropout_rate: the proportion of nodes that will be dropout
        """
        self.params, self.grads = [], []
        self.dropout_rate = dropout_rate
        self.mask = None 
        self.train_flag = True 

    def forward(self, xs): 
        """ Forward propagation of TimeDropout

        Parameter:
            xs: (N * T * D) matrix
        
        Return:
            xs: some elements are set to 0 (i.e. dropout)

        Note:
            Dropout is only used in training process but NOT used in evaluating process.
        """
        if self.train_flag: 
            flag = np.random.randn(*xs.shape) > self.dropout_rate
            scale = 1 / (1.0 - self.dropout_rate)
            self.mask = flag.astype("f") * scale 
            return xs * self.mask 
        else: 
            return xs 
    
    def backward(self, dout):
        """ Backward propagation of TimeDropout

        Parameter:
            dout: a derivative matrix of xs (size: (N * T * D))

        Return:
            dout * mask: (N * T * D) matrix

        Note:
            In the returned matrix, the (elements) nodes that are dropout in 
            forward propagation will also be set to 0.
        """
        return dout * self.mask


class Encoder:
    """ Encoder part (seq2seq)

    """
    def __init__(self, vocab_size, wordvec_size, hidden_size): 
        """ Building Encoder block of seq2seq

        Parameters:
            vocab_size: vocabulary size of a given corpus
            wordvec_size: the length of word vector (embedding layer)
            hidden_size: the length of hidden vector (LSTM layer)

        Matrices and their sizes:
            1. Used in TimeEmbedding layer
                embed_W: (V * D)

            2. Used in TimeLSTM layer
                lstm_Wx: (D * 4H)
                lstm_Wh: (H * 4H)
                lstm_b: a vector with length 4H

        Model:
            xs (N * T) ---> TimeEmbedding layer --- xs (N * T * D) ---> TimeLSTM layer --- hs (N * T * H)
                                                                                            |
                                                                                            |
                                                                                            |
                                                                                            v
                                                                                        hs[:, -1, :] (The last result will be sent to Decode part)
        """
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D) / 100).astype("f")
        lstm_Wx = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H).astype("f")

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)
        
        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None 

    def forward(self, xs): 
        """ Forward propagation of Encoder

        Parameters:
            xs: the input mini-batch with size (N * T)
        
        Return:
            hs[:, -1, :]: the last hidden result will be sent to Decoder part (size: (N * H))
                          (see the comment of __init__ function)
        """
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs 
        return hs[:, -1, :]

    def backward(self, dh):
        """ Back propagation of Encoder

        Parameter: 
            dh: the derivative matrix of hs[:, -1, :] (size: (N * H))
        """
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh 

        dout = self.lstm.backward(dhs)
        self.embed.backward(dout)


class Decoder: 
    """ Decoder part (seq2seq)

    """
    def __init__(self, vocab_size, wordvec_size, hidden_size): 
        """ Initialize Decoder class

        Parameter:
            vocab_size: vocabulary size of a given corpus
            wordvec_size: the length of word vector (embedding layer)
            hidden_size: the length of hidden vector (LSTM layer)

        Matrices and their sizes:
            1. Used in TimeEmbedding layer
                embed_W: (V * D)

            2. Used in TimeLSTM layer
                lstm_Wx: (D * 4H)
                lstm_Wh: (H * 4H)
                lstm_b: a vector with length 4H

            3. Used in Affine layer
                affine_W: (H * V)
                affine_b: a vector with length V

        Model:
            xs (mini-batch of output, size: (N * T)) ---> TimeEmbedding layer --- xs (N * T * D)--|
                                                                                                  |--> TimeLSTM --- hs (N * T * H) ---> TimeAffine ---> score (N * T * V)
                                                                    h (N * H, from Encoder part)--|
        """
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D) / 100).astype("f")
        lstm_Wx = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H).astype("f")
        affine_W = (np.random.randn(H, V) / np.sqrt(H)).astype("f")
        affine_b = np.zeros(V).astype("f")

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], [] 
        for layer in (self.embed, self.lstm, self.affine): 
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, h): 
        """ Forward propagation of Decoder

        Parameters:
            xs: mini-batch of output (size: (N * T))
            h: hidden matrix from Encoder (size: (N * H))

        Return:
            score: score matrix with size (N * T * V)

        Note that we do not calculate loss here. Loss value will be calculate in seq2seq class.
        See Example06 to get more detail.
        """
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)

        return score 
    
    def backward(self, dscore): 
        """ Back propagation of Decoder

        Parameter:
            dscore: the derivative matrix of score (size: (N * T * V))

        Return:
            dh: the derivative matrix of h (size: (N * H)).

        Note that dh will be used in back propagation of Encoder part
        (See backward function of Encoder class)
        """
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        self.embed.backward(dout)
        dh = self.lstm.dh
        return dh

    def generate(self, h, start_id, sample_size):
        """ Generate text automatically with trained model

        Paramters:
            h: hidden matrix (size: (N * H))
            start_id: the start word id
            sample_size: the size that how many words will be selected

        Return:
            sample: an automatically generated text
        """
        sample = [] 
        sample_id = start_id
        self.lstm.set_state(h) 

        for _ in range(sample_size): 
            x = np.array(sample_id).reshape((1, 1)) 
            out = self.embed.forward(x)
            out = self.lstm.forward(out) 
            score = self.affine.forward(out)
            sample_id = np.argmax(score.flatten())
            sample.append(int(sample_id))

        return sample


class PeekyDecoder:
    """ PeekyDecoder is an improved Decoder class

    The difference between Decoder and PeekyDecoder is as follows:
        Decoder: h, the hidden matrix with size (N * H) from Encoder part, 
                 is only used by the first LSTM block.
        
        PeekyDecoder: h is used in all LSTM block and all Affine block

    See this paper to get more detail:

        Kyunghyun Cho, et al: "Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Mochine Translation", 
        arXiv: 1406.1078v3, 2014 (link: https://arxiv.org/abs/1406.1078)
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        """ Initialize

        Parameters:
            vocab_size: vocabulary size of a given corpus
            wordvec_size: the length of word vector (embedding layer)
            hidden_size: the length of hidden vector (LSTM layer)

        Matrices and their sizes:
            1. Used in Embedding layer
                embed_W: (V * D)
            
            2. Used in LSTM layer
                lstm_Wx: ((H + D) * (4H)) 
                lstm_Wh: (H * 4H)
                lstm_b: a vector with length 4H

            3. Used in Affine layer 
                affine_W: ((H + H) * V)
                affine_b: a vector with length V

        Note:
            1. The size of lstm_Wx in Decoder is (D * 4H) while the size of lstm_Wx in PeekyDecoder 
               is ((H + D) * 4H). This is because in PeekyDecoder, we conbime xs and h together as 
               the input of LSTM
            2. h is not only used in LSTM layer but is also used in Affine layer. The size of input 
               of Affine layer in Decoder is (H * V) while it is ((H + H) * V) in PeekyDecoder

        Model:
            The PeekyDecoder model is identical to Decoder model.  
        """
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D) / 100).astype("f")
        lstm_Wx = (np.random.randn(H + D, 4 * H) / np.sqrt(H + D)).astype("f")
        lstm_Wh = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H).astype("f")
        affine_W = (np.random.randn(H + H, V) / np.sqrt(H + H)).astype("f")
        affine_b = np.zeros(V).astype("f")

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], [] 
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None 

    def forward(self, xs, h): 
        """ Forward propagation of PeekyDecoder

        Parameters:
            xs: (N * T) matrix (input mini-batch)
            h: (N * H) matrix (from Encoder part)

        Return:
            score: the score matrix with size (N * T * V)
        """
        N, T = np.shape(xs)
        N, H = np.shape(h)

        self.lstm.set_state(h)
        out = self.embed.forward(xs)
        hs = np.repeat(h, T, axis=0).reshape((N, T, H)) 
        out = np.concatenate((hs, out), axis=2)

        out = self.lstm.forward(out)
        out = np.concatenate((hs, out), axis=2)

        score = self.affine.forward(out)
        self.cache = H
        
        return score 

    def backward(self, dscore):
        """ Back propagation of PeekyDecoder

        Parameter:
            dscore: the derivative matrix of score (size: (N * T * V))

        Return: 
            dh: the derivative matrix of h (size: (N * H))
        """
        H = self.cache

        dout = self.affine.backward(dscore)
        dout, dhs0 = dout[:, :, H:], dout[:, :, :H]
        dout = self.lstm.backward(dout)
        dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
        self.embed.backward(dembed)

        dhs = dhs0 + dhs1
        dh = self.lstm.dh + np.sum(dhs, axis=1)
        return dh

    def generate(self, h, start_id, sample_size): 
        """ Generate text automatically with trained model

        Paramters:
            h: hidden matrix (size: (N * H))
            start_id: the start word id
            sample_size: the size that how many words will be selected

        Return:
            sample: an automatically generated text
        """
        sample = [] 
        char_id = start_id 
        self.lstm.set_state(h)

        H = np.shape(h)[1]
        peeky_h = np.reshape(h, newshape=(1, 1, H)) 
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1, 1))
            
            out = self.embed.forward(x)
            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)
            
            char_id = np.argmax(score.flatten())
            sample.append(char_id)

        return sample


class WeightSum:
    """ WeightSum class (part of Attention)

    WeightSum class is used to calculate the weigh sum of hidden matrix given by LSTM 
    and probability vector a.

    For a single input time series:
        hs: (T * H) hidden matrix given by TimeLSTM layer
        a: a weight vector with length T

        Then, the calculation is as follows:

        a (length T) ----> ar (T * 1) matrix (reshape) ----> t = hs * ar (numpy broadcast) ----> c = sum(t, axis=0) (c is a content vector with lengyth H)

    For a mini-batch:
        hs: (N * T * H)
        a: (N * T)

        The calculation is identical to above one and the result c is an (N * H) content matrix

    """
    def __init__(self):
        """ Initialize WeightSum class

        There is no parameter
        """
        self.params, self.grads = [], []
        self.cache = None 

    def forward(self, hs, a):
        """ Forward propagation of WeightSum class

        Parameters:
            hs: (N * T * H) hidden matrix
            a: (N * T) weight matrix

        Return:
            c: (N * H) content matrix
        """
        N, T, _ = np.shape(hs) 
        
        ar = np.reshape(a, newshape=(N, T, 1))
        t = hs * ar 
        c = np.sum(t, axis=1)

        self.cache = (hs, ar)
        return c

    def backward(self, dc):
        """ Back propagation of WeightSum class

        Parameter:
            dc: the derivative matrix of content matrix c (size: (N * H))

        Return:
            dhs: the derivative matrix of hs (size: (N * T * H))
            da: the derivative matrix of a (size: (N * T))
        """
        hs, ar = self.cache
        N, T, H = np.shape(hs)
        dt = np.reshape(dc, newshape=(N, 1, H))
        dt = np.repeat(dt, repeats=T, axis=1)
        dar = dt * hs
        dhs = dt * ar 
        da = np.sum(dar, axis=2)

        return dhs, da


class AttentionWeight:
    """ AttentionWeight class

    AttentionWeight is used to calculate the weight matrix a.
    Given the matrix hs from TimeLSTM in Encoder and the matrix 
    h from TimeLSTM in Decoder, weight matrix a is calculated by:

    Step 1. Reshape h to hr (size: h (N * H), hr (N * 1 * H))
    Step 2. hr is reshaped to (N * T * H)
    Step 3. t = hr * hs (element-wise product)
    Step 4. s = sum(t, axis=2) (s: (N * T))
    Step 5. a = softmax(s) (convert score s to weight a) (a: (N * T))

    """
    def __init__(self):
        """ Initialize AttentionWeight class

        There is no parameter
        """
        self.params, self.grads = [], [] 
        self.softmax = Softmax()
        self.cache = None 

    def forward(self, hs, h):
        """ Forward propagation of AttentionWeight

        Parameters:
            hs: (N * T * H) matrix
            h: (N * H) matrix

        Return:
            a: (N * T) weight matrix
        """
        N, T, H = np.shape(hs)

        hr = np.reshape(h, newshape=(N, 1, H))
        hr = np.repeat(hr, repeats=T, axis=1)
        t = hs * hr
        s = np.sum(t, axis=2)
        a = self.softmax.forward(s)

        self.cache = (hs, hr)
        return a 

    def backward(self, da):
        """ Back propagation of AttentionWeight class

        Paramter:
            da: the derivative matrix of weight matrix a (size: (N * T))

        Return:
            dhs: the derivative matrix of hs (size: (N * T * H))
            dh: the derivative matrix of h (size: (N * H))
        """
        hs, hr = self.cache
        N, T, H = np.shape(hs)

        ds = self.softmax.backward(da)
        dt = np.reshape(ds, newshape=(N, T, 1))
        dt = np.repeat(dt, repeats=H, axis=2)
        dhr = dt * hs
        dhs = dt * hr
        dh = np.sum(dhr, axis=1)

        return dhs, dh


class Attention:
    """ Single Attention block

    A single Attention block is just constructed by AttentionWeight block and WeightSum block.
    """
    def __init__(self):
        """ Initialize Attention class
        """
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self, hs, h):
        """ Forward propagation of Attention

        Parameters:
            hs: (N * T * H) matrix (from Encoder TimeLSTM block)
            h: (N * H) matrix (from Decoder LSTM block)

        Return:
            out: (N * H) content matrix
        """
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return out

    def backward(self, dout):
        """ Back propagation of Attention

        Parameter:
            dout: the derivative matrix of content matrix out (size: (N * H))

        Return:
            dhs: the derivative matrix of hs (size: (N * T * H))
            dh: the derivative matrix of h (size: (N * H))
        """
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh


class TimeAttention:
    """ TimeAttention class

    TimeAttention class convert single Attention block to time series
    """
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        _, T, _ = np.shape(hs_dec)
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []

        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        return out

    def backward(self, dout):
        _, T, _ = np.shape(dout)
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh

        return dhs_enc, dhs_dec


def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    """ Evaluate perplexity of current model

    Perplexity is used to evaluate a language model. It is defined as

                    perplexity = exp(L)

    where L is the average loss of model.

    The smaller the perplexity is, the better the model is

    Parameters:
        model: a given language model
        corpus: a given corpus 
        batch_size: the batch size
        time_size: time size

    Return:
        ppl: perplexity
    """
    print("evaluating perplexity ... ")
    corpus_size = len(corpus)
    total_loss = 0
    max_iters = (corpus_size - 1) // (batch_size * time_size) 
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters): 
        xs = np.zeros(shape=(batch_size, time_size), dtype=np.int32) 
        ts = np.zeros(shape=(batch_size, time_size), dtype=np.int32) 
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size): 
            for i, offset in enumerate(offsets): 
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]
        
        try:
            loss = model.forward(xs, ts, train_flag=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss 

        sys.stdout.write('\r%d / %d' % (iters, max_iters))
        sys.stdout.flush() 

    print("")
    ppl = np.exp(total_loss / max_iters)
    return ppl 


def eval_seq2seq(model, question, correct, id_to_char, verbos=False, is_reverse=False):
    correct = correct.flatten() 

    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))

    question = "".join([id_to_char[int(c)] for c in question.flatten()])
    correct = "".join([id_to_char[int(c)] for c in correct]) 
    guess = "".join([id_to_char[int(c)] for c in guess])

    if verbos: 
        if is_reverse: 
            question = question[::-1]
        
        colors = {"ok": "\033[92m", "fail": "\033[91m", "close": "\033[0m"}
        print("Q", question)
        print("T", correct) 

        is_windows = os.name == "nt"
        if correct == guess: 
            mark = colors["ok"] + "☑" + colors["close"]
            if is_windows: 
                mark = "O"
            print(mark, " ", guess)
        else: 
            mark = colors["fail"] + "☒" + colors["close"]
            if is_windows:
                mark = "X"
            print(mark, " ", guess)
        print("----")
    
    return 1 if guess == correct else 0