import numpy as np 
import pickle
import sys, os
sys.path.append(os.pardir)
from read_dataset import PennTreeBank
from grad_optimizers import Adam
from trainer import Trainer
from word2vector import CBOW, create_contexts_target

# hyper-parameters
window_size = 5
hidden_size = 100
batch_size = 100 
max_epoch = 10

# load data
ptb = PennTreeBank()
corpus, word_to_id, id_to_word = ptb.load_data(data_type="train")
vocab_size = len(word_to_id)

# create contexts and target
contexts, target = create_contexts_target(corpus, window_size)

# create model
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam() 
trainer = Trainer(model, optimizer)

# training and plot
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# save results
word_vecs = model.word_vecs
params = {}
params["word_vecs"] = word_vecs.astype(np.float16)
params["word_to_id"] = word_to_id
params["id_to_word"] = id_to_word

pkl_file = "cbow_trained_params.pkl"
with open(pkl_file, "wb") as f:
    pickle.dump(params, f, -1)