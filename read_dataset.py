import numpy as np 
import urllib.request
import pickle
import sys, os 


class Spiral:
    """ Generate spiral dataset

    This class is used to generate spiral dataset with
    small gaussian noise.
    """
    def __init__(self):
        pass

    def load_data(self, seed=1984): 
        """ load spiral dataset

        Parameter:
            seed: random number seed. Using fixed seed guarantees 
                  that one can obtain exactly identical dataset everytime.

        Return:
            x: training data (size: 100 * 2, 100: data size, 2: the number of features)
            t: labels (size: 100 * 3, 3: the number of classes). Each row is a one-hot vector with 3 elements.
        """
        np.random.seed(seed)
        N = 100 
        DIM = 2
        CLS_NUM = 3

        x = np.zeros((N * CLS_NUM, DIM)) 
        t = np.zeros((N * CLS_NUM, CLS_NUM), dtype=np.int32)

        for j in range(CLS_NUM): 
            for i in range(N): 
                rate = i / N
                radius = 1.0 * rate
                theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2

                ix = N * j + i 
                x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
                t[ix, j] = 1

        return x, t


class PennTreeBank:
    url_base = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/"
    key_file = {
        "train": "ptb.train.txt",
        "test": "ptb.test.txt",
        "valid": "ptb.valid.txt", 
    }
    save_file = {
        "train": "ptb.train.npy",
        "test": "ptb.test.npy",
        "valid": "ptb.valid.npy", 
    }
    vocab_file = "ptb.vocab.pkl"
    dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "\\dataset"

    def _download(self, file_name): 
        file_path = self.dataset_dir + "\\" + file_name
        if os.path.exists(file_path):
            return

        print("Downloading " + file_name + " ... ")

        try:
            urllib.request.urlretrieve(self.url_base + file_name, file_path)
        except urllib.error.URLError:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib.request.urlretrieve(self.url_base + file_name, file_path)

        print("Done")

    def load_vocab(self):
        vocab_path = self.dataset_dir + "\\" + self.vocab_file

        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                word_to_id, id_to_word = pickle.load(f)
            return word_to_id, id_to_word

        word_to_id = {}
        id_to_word = {}
        data_type = "train"
        file_name = self.key_file[data_type]
        file_path = self.dataset_dir + "\\" + file_name

        self._download(file_name)

        words = open(file_path).read().replace("\n", "<eos>").strip().split()

        for _, word in enumerate(words):
            if word not in word_to_id:
                tmp_id = len(word_to_id)
                word_to_id[word] = tmp_id
                id_to_word[tmp_id] = word

        with open(vocab_path, "wb") as f:
            pickle.dump((word_to_id, id_to_word), f)

        return word_to_id, id_to_word

    def load_data(self, data_type="train"):
        if data_type == "val": data_type = "valid"
        save_path = self.dataset_dir + "\\" + self.save_file[data_type]

        word_to_id, id_to_word = self.load_vocab()

        if os.path.exists(save_path):
            corpus = np.load(save_path)
            return corpus, word_to_id, id_to_word

        file_name = self.key_file[data_type]
        file_path = self.dataset_dir + "\\" + file_name
        self._download(file_name)

        words = open(file_path).read().replace("\n", "<eos>").strip().split()
        corpus = np.array([word_to_id[w] for w in words])

        np.save(save_path, corpus)
        return corpus, word_to_id, id_to_word


class Sequence:
    def __init__(self):
        self.id_to_char = {} 
        self.char_to_id = {}
    
    def update_vocab(self, txt):
        chars = list(txt)
        for _, char in enumerate(chars):
            if char not in self.char_to_id:
                tmp_id = len(self.char_to_id)
                self.char_to_id[char] = tmp_id
                self.id_to_char[tmp_id] = char

    def load_data(self, file_name="addition.txt", seed=1984):
        file_path = os.path.dirname(os.path.abspath(__file__)) + "\\dataset\\" + file_name

        if not os.path.exists(file_path):
            print("No file: %s" % file_name)
            return None

        questions, answers = [], []

        for line in open(file_path, "r"):
            idx = line.find("_")
            questions.append(line[:idx])
            answers.append(line[idx:-1])

        # create vocab dict
        for i in range(len(questions)):
            q, a = questions[i], answers[i]
            self.update_vocab(q)
            self.update_vocab(a)

        # create numpy array
        x = np.zeros((len(questions), len(questions[0])), dtype=np.int)
        t = np.zeros((len(questions), len(answers[0])), dtype=np.int)

        for i, sentence in enumerate(questions):
            x[i] = [self.char_to_id[c] for c in list(sentence)]
        for i, sentence in enumerate(answers):
            t[i] = [self.char_to_id[c] for c in list(sentence)]

        # shuffle
        indices = np.arange(len(x))
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
        x = x[indices]
        t = t[indices]

        # 10% for validation set
        split_at = len(x) - len(x) // 10
        (x_train, x_test) = x[:split_at], x[split_at:]
        (t_train, t_test) = t[:split_at], t[split_at:]

        return (x_train, t_train), (x_test, t_test)

    def get_vocab(self):
        return self.char_to_id, self.id_to_char