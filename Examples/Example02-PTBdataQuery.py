import numpy as np 
import sys, os
from sklearn.utils.extmath import randomized_svd
sys.path.append(os.pardir)
from read_dataset import PennTreeBank
from word2vector import create_co_matrix, ppmi, most_similar

window_size = 2
wordvec_size = 100 

ptb = PennTreeBank()
corpus, word_to_id, id_to_word = ptb.load_data(data_type="train")
vocab_size = len(word_to_id)
print("counting co-ocurrence ...")
C = create_co_matrix(corpus, vocab_size, window_size=window_size)
print("calculating PPMT ...")
W = ppmi(C, verbose=True)

print("calculating SVD ...")
# We use SVD (Singular Value Decomposition) to perform dimensionality reduction
U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)

word_vecs = U[:, :wordvec_size]

querys = ["you", "year", "car", "toyota"]
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
