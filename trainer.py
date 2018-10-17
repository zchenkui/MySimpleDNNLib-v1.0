import numpy as np 
import time
import matplotlib.pyplot as plt 
from basic_functions import remove_duplicate, clip_grads


class Trainer:
    """ General forward neural networks training framework

    Given a neural network model and an optimizer, this class will 
    training the model.
    """
    def __init__(self, model, optimizer):
        """ Initialize Trainer class

        Parameters:
            model: a neural network model
            optimizer: an optimizer algorithm
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0 

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20, verbose=True): 
        """ fit method

        This method is used to train the model

        Parameters:
            x: training data
            t: training label
            max_epoch: max epoch, default 10
            batch_size: batch size, default 32 
            max_grad: gradient threshold
            eval_interval: evaluation interval, default 20
            verbose: print out the training detail or not

        Result:
            None
        """
        data_size = len(x)
        max_iters = data_size // batch_size # max iterations used to run one epoch
        self.eval_interval = eval_interval
        model = self.model
        optimizer = self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()    # record the start time
        print("start training ... ")
        for _ in range(max_epoch): # for each epoch
            idx = np.random.permutation(data_size)  # shuffle data
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                # create data and label batch
                x_batch = x[iters * batch_size : (iters + 1) * batch_size]
                t_batch = t[iters * batch_size : (iters + 1) * batch_size]
                
                loss = model.forward(x_batch, t_batch)  # forward propagation
                model.backward()    # back propagation
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_norm=max_grad)
                optimizer.update(params, grads) # update weights
                total_loss += loss 
                loss_count += 1

                if (eval_interval is not None) and (iters % eval_interval == 0): 
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    if verbose: 
                        print("| epoch %d |  iter %d / %d | time %d[s] | loss %.2f" 
                            % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss = 0
                    loss_count = 0

            self.current_epoch += 1
        print("end training ... ")
        print() 

    def plot(self, ylim=None):
        """ plot loss list
        """
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel("iterations (x" + str(self.eval_interval) + ")")
        plt.ylabel("loss")
        plt.show()


class RnnlmTrainer:
    """ Train a RNN language model (Rnnlm)

    The difference between Trainer class and RnnlmTrainer class is that the former will shuffle 
    the training data and the latter will not because the training data used in RnnlmTrainer class 
    is time series sequence data whose order should be kept. 
    """
    def __init__(self, model, optimizer):
        """ Initialize RnnlmTrainer class 

        Parameters:
            model: a RNN language model 
            optimizer: an optimizer algorithm
        """
        self.model = model 
        self.optimizer = optimizer 
        self.time_idx = None 
        self.ppl_list = None 
        self.eval_interval = None 
        self.current_epoch = 0

    def get_batch(self, x, t, batch_size, time_size): 
        """ Create a mini-batch for training data and the labels

        Parameters: 
            x: training sequence data
            t: the labels of training data
            batch_size: batch size
            time_size: the number of words in one batch

        Return: 
            batch_x: mini-batch of training data
            batch_t: the labels of mini-batch

        Example:
            Let:
                corpus = "a b c d e f g h i j k l m n o p q r s t u v w x y z a b c d e" (length: 31)
                x = "a b c d e f g h i j k l m n o p q r s t u v w x y z a b c d" (length: 30)
                t = "b c d e f g h i j k l m n o p q r s t u v w x y z a b c d e" (length: 30)
                batch_size = 3
                time_size = 5
            
            Then:
                data_size = 30 
                jump = floor(data_size / batch_size) = 10
                offsets = [0, 10, 15]
            
            The batches generated in the first two epoches are:
            EPOCH 1:
                Loop 1:
                    batch_x_1 = 
                    [
                        batch_1: [a b c d e], 
                        batch_2: [k l m n o], 
                        batch_3: [u v w x y], 
                    ]
                    batch_t_1 = 
                    [
                        label_1: [b c d e f], 
                        label_2: [l m n o p], 
                        label_3: [v w x y z], 
                    ]

                Loop 2: 
                    batch_x_2 = 
                    [
                        batch_1: [f g h i j], 
                        batch_2: [p q r s t], 
                        batch_3: [z a b c d],
                    ]
                    batch_t_2 = 
                    [
                        label_1: [g h i j k], 
                        label_2: [q r s t u], 
                        label_3: [a b c d e],
                    ]
            
            EPOCH 2:
                Loop 1:
                    batch_x_1 = 
                    [
                        batch_1: [k l m n o], 
                        batch_2: [u v w x y], 
                        batch_3: [a b c d e],
                    ]
                    batch_t_1 = 
                    [
                        label_1: [l m n o p], 
                        label_2: [v w x y z], 
                        label_3: [b c d e f],
                    ]

                Loop 2:
                    batch_x_2 = 
                    [
                        batch_1: [p q r s t], 
                        batch_2: [z a b c d], 
                        batch_3: [f g h i j], 
                    ]
                    batch_t_2 = 
                    [
                        label_1: [q r s t u], 
                        label_2: [a b c d e], 
                        label_3: [g h i j k],
                    ]

        """
        batch_x = np.empty((batch_size, time_size), dtype="i")
        batch_t = np.empty((batch_size, time_size), dtype="i") 

        data_size = len(x) 
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1
        
        return batch_x, batch_t

    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35, max_grad=None, eval_interval=20, verbose=True): 
        """ Train Rnnlm model

        Parameters:
            xs: training sequence data
            ts: labels of xs
            max_epoch: max epoch, default 10
            batch_size: batch size, default 20 
            time_size: time size, default 35
            max_grad: gradient threshold
            eval_interval: evaluation interval, default 20
            verbose: print out the training detail or not
        """
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size) 
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model = self.model
        optimizer = self.optimizer
        total_loss = 0
        loss_count = 0 

        start_time = time.time() 
        print("start training ...")
        for _ in range(max_epoch): 
            for iters in range(max_iters): 
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                loss = model.forward(batch_x, batch_t)
                model.backward() 
                params, grads = remove_duplicate(model.params, model.grads) 
                if max_grad is not None: 
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss 
                loss_count += 1 

                if (eval_interval is not None) and (iters % eval_interval) == 0: 
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time 
                    if verbose: 
                        print("| epoch %d |  iter %d / %d | time %d[s] | perplexity %.2f"
                            % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1
        print("end training ...")
        print() 

    def plot(self, ylim=None): 
        """ Plot the perplexity of training
        """
        x = np.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('perplexity')
        plt.show()