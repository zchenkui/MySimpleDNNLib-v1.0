import numpy as np 
import sys, os
sys.path.append(os.pardir)
from time_series import TimeEmbedding, TimeRNN, TimeAffine, TimeSoftmaxWithLoss
from grad_optimizers import SGD
from trainer import RnnlmTrainer
from read_dataset import PennTreeBank


class SimpleRnnlm:
    """ Building a simple RNN language model 

    """
    def __init__(self, vocab_size, wordvec_size, hidden_size): 
        """ Initialize SimpleRnnlm class

        Parameters:
            vocab_size: vocabulary size
            wordvec_size: embedding size 
            hidden_size: RNN hidden size

        Matrices' size: 
            embed_W: (V * D)
            rnn_Wx: (D * H)
            rnn_Wh: (H * H)
            rbb_b: H (a vector)
            affine_W: (H * V)
            affine_b: V (a vector)

        Structure:
            training data --> TimeEmbedding layer --> TimeRNN layer --> TimeAffine layer --> TimeSoftmaxWithLoss layer --> loss
        """
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D) / 100).astype("f")
        rnn_Wx = (np.random.randn(D, H) / np.sqrt(D)).astype("f")
        rnn_Wh = (np.random.randn(H, H) / np.sqrt(H)).astype("f")
        rnn_b = np.zeros(H).astype("f")
        affine_W = (np.random.randn(H, V) / np.sqrt(H)).astype("f")
        affine_b = np.zeros(V).astype("f")

        self.layers = [
            TimeEmbedding(embed_W), 
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True), 
            TimeAffine(affine_W, affine_b), 
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        self.params, self.grads = [], [] 
        for layer in self.layers: 
            self.params += layer.params 
            self.grads += layer.grads

    def forward(self, xs, ts): 
        """ Forward propagation of SimpleRnnlm model

        Parameters:
            xs: training data
            ts: labels of training data

        Return:
            loss
        """
        for layer in self.layers: 
            xs = layer.forward(xs) 
        loss = self.loss_layer.forward(xs, ts)
        return loss 

    def backward(self, dout=1): 
        """ Backward propagation of SimpleRnnlm model

        Parameter:
            dout: the derivative of loss (a scalar)
        """
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def reset_state(self):
        """ Reset state of RNN
        """
        self.rnn_layer.reset_state()


np.random.seed(seed=7)

batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5
lr = 0.1
max_epoch = 100

# read ptb data and prepare training data
ptb = PennTreeBank()
corpus, word_to_id, id_to_word = ptb.load_data()
corpus_size = 1000
corpus = corpus[0 : corpus_size]
vocab_size = int(max(corpus) + 1) 
xs = corpus[:-1]
ts = corpus[1:]

# build RNN language model
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# training and plot
trainer.fit(xs, ts, max_epoch, batch_size, time_size)
trainer.plot()