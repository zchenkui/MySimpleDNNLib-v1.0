import numpy as np 
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.pardir)
from time_series import Encoder, Decoder, PeekyDecoder, TimeSoftmaxWithLoss, eval_seq2seq
from read_dataset import Sequence
from trainer import Trainer
from grad_optimizers import Adam


class Seq2Seq:
    """ Seq2Seq Model

    This is the base-line of seq2seq model.

                    Loss
                     ^
                     |
                     |
                     |
        TimeSoftmaxWithLoss layer
                     ^
                     |
                     |
                     |
    Encoder ----> Decoder

    See the comment of Encoder and Decoder classes to get more detail
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        """ Initialize Seq2Seq class

        Parameters:
            vocab_size: vocabulary size of a given corpus
            wordvec_size: the length of word vector (embedding layer)
            hidden_size: the length of hidden vector (LSTM layer)
        """
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H) 
        self.softmax = TimeSoftmaxWithLoss() 

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        """ Forward propagation of Seq2Seq

        Parameters:
            xs: mini-batch with size (N * T)
            ts: labels with size (N * T)

        Return:
            loss
        """
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]
        h = self.encoder.forward(xs) 
        score = self.decoder.forward(decoder_xs, h) 
        loss = self.softmax.forward(score, decoder_ts) 
        return loss 

    def backward(self, dout=1):
        """ Back propagation of Seq2Seq

        Parameters:
            dout: the derivative of loss (a scalar)
        """
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        self.encoder.backward(dh)

    def generate(self, xs, start_id, sample_size): 
        """ Generate text automatically
        """
        h = self.encoder.forward(xs)
        sample = self.decoder.generate(h, start_id, sample_size)
        return sample 


class PeekySeq2Seq(Seq2Seq):
    """ Seq2Seq Model

    This is the peeky seq2seq model.

                    Loss
                     ^
                     |
                     |
                     |
        TimeSoftmaxWithLoss layer
                     ^
                     |
                     |
                     |
    Encoder ----> PeekyDecoder

    See the comment of Encoder and PeekyDecoder classes to get more detail
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size): 
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H) 
        self.softmax = TimeSoftmaxWithLoss() 

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

sq = Sequence()
(x_train, t_train), (x_test, t_test) = sq.load_data()
char_to_id, id_to_char = sq.get_vocab()

is_reversed = True
if is_reversed:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

#model = Seq2Seq(vocab_size, wordvec_size, hidden_size)
model = PeekySeq2Seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose, is_reversed)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc * 100))

x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.show()