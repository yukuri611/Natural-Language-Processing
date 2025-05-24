import sys
sys.path.append('./')

import numpy as np
from common.util import preprocess, create_co_matrix, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(id_to_word)
co_matrix = create_co_matrix(corpus, vocab_size, 1)

M = ppmi(co_matrix, False)
U,S,V = np.linalg.svd(M)