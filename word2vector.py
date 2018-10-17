import numpy as np 
import collections
from basic_layers import SigmoidWithLoss


def preprocess(text):
    """ Converting an article to corpus and generating id for each word in it

    Parameter: 
        text: (type: string) a given article

    Return:
        corpus: (type: numpy.array) having the same content with text, but the words are replaced by their id.
        word_to_id: (type: dictionary) a dictionary whose ("key", "value") is ("word", "id").
        id_to_word: (type: dictionary) a dictionary whose ("value", "key") is ("id", "word").

    Example:
        Input text: "You say goodbye and I say hello."
        Output:
            corpus: numpy.array([0, 1, 2, 3, 4, 1, 5, 6])
            word_to_id: {"you": 0, "say": 1, "goodbye": 2, "and": 3, "i": 4, "hello": 5, ".": 6}
            id_to_word: {0: "you", 1: "say", 2: "goodbye", 3: "and", 4: "i", 5: "hello", 6: "."}
    """
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")

    word_to_id, id_to_word = {}, {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    """ Create co-occurence matrix

    Given a corpus in which the number of vocabulary is vocab_size, this function 
    creates a (vocab_size * vocab_size) co-occurence matrix according to a window_size 
    predefined by user. Each row of co-occurence matrix is a vector that records 
    appearing times of words at the neighbour of current word.

    Parameters:
        corpus, vocab_size, and window_size

    Return:
        co_matrix: co-occurence matrix

    Example:
        Given a text: "You say goodbye and I say hello.", we can convert it to corpus 
        and word_to_id:

        corpus: numpy.array([0, 1, 2, 3, 4, 1, 5, 6])
        word_to_id: {"you": 0, "say": 1, "goodbye": 2, "and": 3, "i": 4, "hello": 5, ".": 6}

        So, the vocab_size is 7. If window_size = 2, the co_matrix is:

        [
            [0, 1, 1, 0, 0, 0, 0], 
            [1, 0, 1, 2, 1, 1, 1], 
            [1, 1, 0, 1, 1, 0, 0], 
            [0, 2, 1, 0, 1, 0, 0], 
            [0, 1, 1, 1, 0, 1, 0], 
            [0, 1, 0, 0, 1, 0, 1], 
            [0, 1, 0, 0, 0, 1, 0],
        ],

        for example, the 3rd row [1, 1, 0, 1, 1, 0, 0] is corresponding to the word "goodbye". 
        Given a window_size 2, this vector shows that at the neighbour (here, 2) of "goodbye", 
        the words "you", "say", "and", and "I" are appeared one time respectively.

        Note: vocab_size is NOT the length of corpus.
    """
    corpus_size = len(corpus) 
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus): 
        for i in range(1, window_size + 1):
            left_idx = idx - i 
            right_idx = idx + i 

            if left_idx >= 0: 
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if right_idx < corpus_size: 
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    """ Calculate the cosine similarity between vectors x and y

    cos_similarity(x, y) = (x.y) / (||x|| * ||y||)
    """
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


def ppmi(C, verbose=False, eps=1e-7): 
    """ Calculate Positive Pointwise Mutual Information  (PPMI) matrix from co-occurence matrix

    Co-occurence matrix just records the appearing times of neighbour words, while PPMI matrix 
    denotes the correlation among words. See below reference to get more information about PMI:

    https://en.wikipedia.org/wiki/Pointwise_mutual_information
    
    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = np.shape(C)[0] * np.shape(C)[1]
    cnt = 0 

    for i in range(np.shape(C)[0]): 
        for j in range(np.shape(C)[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose: 
                cnt += 1
                if cnt % (total // 100) == 0: 
                    print("%.1f%% done" % (100 * cnt / total))
    
    return M 


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    """ Search the most similar words

    Given a PPMI matrix and a target word, find out the top most similar words 
    from vocabulary according to the cosine similarity.

    Parameters:
        query: the target word
        word_to_id: dictionary of (word, id), where key = word and value = id
        id_to_word: dictionary of (id, word), where key = id and value = word
        word_matrix: PPMI matrix
        top: the top most similar words will be output
    
    Return:
        None

    Print:
        the top most similar words
    """
    if query not in word_to_id:
        print("%s is not found" % query)
        return 

    print("\n[query] ", query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size): 
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0 
    for i in np.argsort(-1 * similarity):
        if id_to_word[i] == query:
            continue
        print(" %s: %f" % (id_to_word[i], similarity[i]))
        count += 1
        if count >= top:
            break 


def create_contexts_target(corpus, window_size=1):
    """ Create contexts and targets from corpus

    Parameters:
        corpus: a training corpus with word ids
        window_size: the half of context words, i.e. the number of context words = window_size * 2

    Return:
        contexts: a context matrix with size = ((length of corpus - 2 * window_size) * (window_size * 2))
        target: a vector with (length of corpus - 2 * window_size) elements
    """
    target = corpus[window_size : -window_size]
    contexts = [] 

    for idx in range(window_size, len(corpus) - window_size): 
        cs = [] 
        for t in range(-window_size, window_size + 1): 
            if t == 0: 
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target) 


class Embedding: 
    """ Embedding layer

    Convert a "word" to a vector with fixed length
    """
    def __init__(self, W): 
        """ Initialize Embedding class

        Parameter:
            W: embedding matrix (size: V * H, V: the number of vocabulary, H: the length of hidden vector)
        """
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None 

    def forward(self, idx): 
        """ Forward propagation of Embedding class

        Parameter:
            idx: a list of ids of words

        Return:
            out: a matrix where each row is the corresponding row of self.W

        Save: 
            idx
        """
        W, = self.params
        self.idx = idx 
        out = W[idx]
        return out 

    def backward(self, dout): 
        """ Backward propagation of Embedding class

        Add each row of dout to the corresponding row of dW according to self.idx, 
        the size of dW is V * H, which is identical to W

        Parameter:
            dout: a derivative matrix (size: I * H, where I is the length of self.idx and H is the length of hidden vector)
                  from above layer

        Return:
            None 
        """
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)


class EmbeddingDot: 
    """ EmbeddingDot class is one of key parts of NegativeSampling algorithm

    Without performing matrix product, EmbeddingDot class just selects the target row from 
    Embedding matrix and perform dot product so that it has better performance than naive wrod2vec algorithm.
    """
    def __init__(self, W):
        """ Initialize EmbeddingDot class

        Parameter: 
            W: an embedding matrix with size (V * H) where V is the number of vocabulary and H is the length of hidden vector
        """
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads 
        self.cache = None 

    def forward(self, h, idx): 
        """ Forward propagation of EmbeddingDot class

        Parameter:
            h: a matrix with size (N * H) where N is the batch size and H is the length of hidden vector.
            idx: a vector with N word ids and each id identifies the target word to a sample.

        Return:
            out: a "score" vector with N elements

        Save:
            save h and target_W to self.cache, where target_W is a matrix with size (N * H)
        """
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out 

    def backward(self, dout): 
        """ Back-propagation of EmbeddingDot class

        Parameter:
            dout: a derivative vector with N elements. dout is a derivative vector corresponding to the score vector

        Return:
            dh: a derivative matrix with size (N * H), which is corresponding to h in forward function
        """
        h, target_W = self.cache
        dout = np.reshape(dout, newshape=(np.shape(dout)[0], 1)) # convert vector dout to a (N * 1) matrix

        dtarget_W = dout * h 
        self.embed.backward(dtarget_W)
        dh = dout * target_W

        return dh


class UnigramSampler:
    """ Sample some "NEGATIVE" (i.e. wrong) samples from training dataset

    This class randomly selects several negative samples from training dataset.
    Negative samples are those samples whose labels are NOT the target label.

    UnigramSampler class is one of the key parts of NegativeSampling algorithm.
    """
    def __init__(self, corpus, power, sample_size): 
        """ Initialize UnigramSampler class

        The frequencies of words will also be calculated in initialization.

        Parameters:
            corpus: a corpus used to train
            power: a scale number used in sampling
            sample_size: the number of samples that should be selected
        """
        self.sample_size = sample_size
        self.vocab_size = None 
        self.word_p = None 

        counts = collections.Counter() 
        for word_id in corpus: 
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        # scaling by power before calculating the word probabilities
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target): 
        """ Perform negative sampling

        Parameter:
            target: the target sample
        
        Return:
            negative_sample: a matrix with size (batch_size * self.sample_size). Each row has self.sample_size elements
                             that are ids of samples whose labels are different from the label of target.
        """
        batch_size = np.shape(target)[0]
        
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
        for i in range(batch_size): 
            p = self.word_p.copy() 
            target_idx = target[i]
            p[target_idx] = 0
            p /= np.sum(p)
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        
        # This version has better performance but may select the target
        # negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size), replace=False, p=self.word_p)

        return negative_sample


class NegativeSamplingLoss: 
    """ Combine negative sampling and loss layer

    Given a batch (N * H) from hidden layer, NegativeSamplingLoss first performs negative sampling 
    and calculates the loss of the batch.
    """
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        """ Initialize NegativeSamplingLoss class

        Parameters:
            W: an embedding matrix with size (V * H), where V is the number of vocabulary and H is the length of hidden vector
            corpus: a training corpus containing word ids
            power: a scale used in negative sampling algorithm (default 0.75)
            sample_size: the number of negative samples
        """
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params = []
        self.grads = [] 
        for layer in self.embed_dot_layers: 
            self.params += layer.params
            self.grads += layer.grads 

    def forward(self, h, target): 
        """ Forward propagation of NegativeSamplingLoss class

        Parameters:
            h: a hidden batch matrix with size (N * H) where N is the batch size and H is the length of hidden vector
            target: a target matrix with size (N * 1) and each element is the target word id of the corresponding sample

        Return: 
            loss: the loss of h
        """
        batch_size = np.shape(target)[0] 
        negative_sample = self.sampler.get_negative_sample(target)

        # correct label
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # error label 
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size): 
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[i + 1].forward(h, negative_target)
            loss += self.loss_layers[i + 1].forward(score, negative_label)

        return loss 

    def backward(self, dout=1): 
        """ Back propagation of NegativeSamplingLoss

        Parameter:
            dout: a scalar that is the derivative of loss

        Return: 
            dh: a derivative matrix of h, which has the identical size with h (N * H)
        """
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers): 
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        return dh


class CBOW: 
    """ Continue Bag of Words (CBOW) model

    CBOW is a model that obtains the relationship among words in a corpus.

    Model:
        Embedding layer ---|
        Embedding layer ---|
        .
        .               ---|--> h (N * H) --> NegativeSamplingLoss layer --> loss (scalar)
        .
        .
        Embedding layer ---|
    """
    def __init__(self, vocab_size, hidden_size, window_size, corpus): 
        """ Initialize CBOW model

        Parameters:
            vocab_size: the size of vocabulary
            hidden_size: the number of nodes of hidden layer
            window_size: the context size of a target. For example: if corpus = "abcde", target = "c", and window_size = 2, 
                         then the context words are "a", "b" (left two of "c") and "d", "e" (right two of "c")
            corpus: a training corpus with word id.
        """
        V, H = vocab_size, hidden_size

        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(V, H).astype("f")

        self.in_layers = []
        for _ in range(window_size * 2): 
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        layers = self.in_layers + [self.ns_loss]
        self.params = [] 
        self.grads = [] 
        for layer in layers: 
            self.params += layer.params
            self.grads += layer.grads
        
        self.word_vecs = W_in

    def forward(self, contexts, target): 
        """ Forward propagation of CBOW

        Parameters:
            contexts: a contexts matrix with size (N * (window_size * 2)) where N is the batch size
            targets: a vector with N elements and each element is the corresponding target word id

        Return:
            loss: the loss the mini-batch
        """
        h = 0
        for i, layer in enumerate(self.in_layers): 
            h += layer.forward(contexts[:, i])
        h /= len(self.in_layers)    # h is an (N * H) matrix
        loss = self.ns_loss.forward(h, target)
        return loss 

    def backward(self, dout=1):
        """ Back propagation of CBOW

        Parameter:
            dout: the derivative of the loss (default: 1)

        Return: 
            None
        """ 
        dout = self.ns_loss.backward(dout)
        dout /= len(self.in_layers)
        for layer in self.in_layers: 
            layer.backward(dout)