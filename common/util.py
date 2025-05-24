import numpy as np

def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    word_list = text.split(' ')

    corpus = []
    word_to_id = {}
    id_to_word = {}
    next_id = 0
    for word in word_list:
        if word not in word_to_id:
            word_to_id[word] = next_id
            id_to_word[next_id] = word
            next_id += 1
        corpus.append(word_to_id[word])

    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size):
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for i, word_id in enumerate(corpus):
        for w in range(1, window_size + 1):
            l = i - w
            r = i + w
            if l >= 0:
                left_word_id = corpus[l]
                co_matrix[word_id, left_word_id] += 1
            
            if r < len(corpus):
                right_word_id = corpus[r]
                co_matrix[word_id, right_word_id] += 1
            
    return co_matrix


def ppmi(C, verbose, eps = 1e-8):
    #positive pmi
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)

    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i,j] = max(pmi, 0)

            if verbose:
                cnt += 1
                if cnt % (total // 10) == 0:
                    print(f'{round(cnt * 100 / total, 0)}% done')
    
    return M
            
    


