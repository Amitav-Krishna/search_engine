def one_hot_encode(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res
