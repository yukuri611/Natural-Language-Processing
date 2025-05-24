import sys
sys.path.append('./')

import numpy as np
import time
from common.util import create_co_matrix, ppmi
from dataset import ptb


window_size = 2
wordvec_size = 100

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(id_to_word)
co_matrix = create_co_matrix(corpus, vocab_size, window_size)

print('calculating PPMI ...')
start = time.time()
M = ppmi(co_matrix, True)
end = time.time()
print(f"共起行列をPPMIに変換するのにかかった時間: {end-start:.6} sec\n")

print('calculating SVD ...')
start = time.time()
try:
    # truncated SVD (fast!)
    print("using sklearn")
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(M, n_components=wordvec_size, n_iter=5,
                             random_state=None)
    
except ImportError:
    # SVD (slow)
    U, S, V = np.linalg.svd(M)
end = time.time()
print(f"SVDを求めるのにかかった時間: {end-start:.6} sec\n")


word_vecs = U[:, :wordvec_size]
