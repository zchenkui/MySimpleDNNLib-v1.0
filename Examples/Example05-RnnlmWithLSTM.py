import numpy as np 
import pickle
import sys, os 
sys.path.append(os.pardir)
from time_series import TimeEmbedding, TimeDropout, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss, eval_perplexity
from grad_optimizers import SGD
from trainer import RnnlmTrainer
from read_dataset import PennTreeBank


class RnnlmWithLSTM:
    """ RNN language model with LSTM
    """
    def __init__(self, vocab_size=10000, wordvec_size=650, hidden_size=650, dropout_rate=0.5):
        """ Initialize RnnlmWithLSTM

        We created a 8 layers deep RNN language model with LSTM

        Parameters:
            vocab_size: the vocabulary size of a given corpus
            wordvec_size: the length of word vector (used in embedding layer)
            hidden_size: the length of output vector of LSTM layer
            dropuout_rate: the proportion of nodes that will be dropout in Dropout layer

        Matrices and their sizes:
            1. Used in embedding layer:
                embed_W: (V * D)

            2. Used in the first LSTM layer:
                lstm_Wx1: (D * 4H)
                lstm_Wh1: (H * 4H)
                lstm_b1: a vector with length 4H

            3. Used in the second LSTM layer:
                lstm_Wx2: (H * 4H) 
                lstm_Wh2: (H * 4H)
                lstm_b2: a vector with length 4H

            4. Used in Affine layer:
                transpose(embed_W): (D * V)
                affine_b: a vector with length V

            Note:
                1. the size of lstm_Wx2 is different from that of lstm_Wx1
                2. embed_W is used by both embedding layer and affine layer.

        Model:
            xs (input mini-batch (N * T)) ---> TimeEmbedding layer --- xs (N * T * D) ---> TimeDropout layer 1 
            --- xs (dropout xs (N * T * D)) ---> TimeLSTM layer 1 --- xs (N * T * H) ---> TimeDropout layer 2
            --- xs (dropout xs (N * T * H)) ---> TimeLSTM layer 2 --- xs (N * T * H) ---> TimeDropout layer 3 
            --- xs (dropout xs (N * T * H)) ---> TimeAffine layer --- xs (N * T * V) ---> TimeSoftmaxWithLoss 
            ---> loss (a scalar)
        """
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D) / 100).astype("f")
        lstm_Wx1 = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh1 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b1 = np.zeros(4 * H).astype("f") 
        lstm_Wx2 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_Wh2 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b2 = np.zeros(4 * H).astype("f") 
        affine_b = np.zeros(V).astype("f") 

        self.layers = [
            TimeEmbedding(embed_W), 
            TimeDropout(dropout_rate), 
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True), 
            TimeDropout(dropout_rate), 
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True), 
            TimeDropout(dropout_rate), 
            TimeAffine(np.transpose(embed_W), affine_b), 
        ]
        self.loss_layer = TimeSoftmaxWithLoss() 
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        self.params, self.grads = [], [] 
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs, train_flag=False):
        """ given an input mini-batch xs, predict function outputs the related probabilities of xs
        """
        for layer in self.drop_layers:
            layer.train_flag = train_flag

        for layer in self.layers:
            xs = layer.forward(xs)

        return xs 

    def forward(self, xs, ts, train_flag=True):
        """ Forward propagation of RnnlmWithLSTM

        Paramters:
            xs: the input mini-batch (size: N * T)
            ts: the labels of xs (size: N * T)

        Return:
            loss: the loss of current model
        """
        score = self.predict(xs, train_flag)
        loss = self.loss_layer.forward(score, ts) 
        return loss 

    def backward(self, dout=1):
        """ Back propagation of RnnlmWithLSTM

        Parameter:
            dout: the derivative of loss (a scalar)

        Return:
            dout: the derivative matrix of input mini-batch xs (size: N * T)
        """
        dout=self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout) 
        return dout 

    def reset_state(self): 
        """ Reset LSTM state to None
        """
        for layer in self.lstm_layers:
            layer.reset_state() 

    def save_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        params = [p.astype(np.float16) for p in self.params]

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)


# hyper parameters
batch_size = 20
wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
max_epoch = 40
max_grad = 0.25
dropout = 0.5

# load data
ptb = PennTreeBank()
corpus, word_to_id, id_to_word = ptb.load_data(data_type="train")
corpus_val, _, _ = ptb.load_data(data_type="val")
corpus_test, _, _ = ptb.load_data(data_type="test")

# create xs and ts
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# build model
model = RnnlmWithLSTM(vocab_size, wordvec_size, hidden_size, dropout)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# train model and save the best parameters
best_ppl = float("inf")
for epoch in range(max_epoch):
    trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size,
                time_size=time_size, max_grad=max_grad)

    model.reset_state()
    ppl = eval_perplexity(model, corpus_val)
    print("valid perplexity: ", ppl)

    if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()
    else:
        lr /= 4.0
        optimizer.lr = lr

    model.reset_state()
    print('-' * 50)

# After training, we evaluate the trained model on test data
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('test perplexity: ', ppl_test)