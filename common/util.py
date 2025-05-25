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
            
    


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size: -window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size+1):
            if t == 0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)
    
    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    '''one-hot表現への変換

    :param corpus: 単語IDのリスト（1次元もしくは2次元のNumPy配列）
    :param vocab_size: 語彙数
    :return: one-hot表現（2次元もしくは3次元のNumPy配列）
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot