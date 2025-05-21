def generate_training_data(tokens, word_to_id, window=2): # Takes the list of tokens, the word to id dictionary, and the maximum distance between the centre word and context word.
    import one_hot_encode
    from mapping import id_to_word
    from human_readable_vectors import human_readable_vectors

    X = [] # stores one_hot_encoded centre words
    y = [] # stores one hot encoded context words
    n_tokens = len(tokens) # the number of tokens

    id_to_word_map = id_to_word(tokens)

    for i in range(n_tokens):
        left = range(max(0, i - window), i) # A range of numbers to the specified distance to the left (or until the beginning, if the specified distance to the left would go out of bounds).
        right = range(i+1, min(n_tokens, i + window + 1)) # A range of numbers to the specified distance to the right (or until the beginning, if the specified distance to the right would go out of bounds).

        for j in list(left) + list(right): # Loops through the tokens to the right or left (context tokens)
          X.append(one_hot_encode.one_hot_encode(word_to_id[tokens[i]], len(word_to_id)))
          y.append(one_hot_encode.one_hot_encode(word_to_id[tokens[j]], len(word_to_id)))
          print(human_readable_vectors(X[-1], y[-1], id_to_word_map))

    return X, y

if __name__ == "__main__":
    import tokenization
    from mapping import word_to_id, id_to_word
    sentence = input("Give me the sentence: ")
    tokens = tokenization.tokenization(sentence)
    word_to_id, id_to_word = word_to_id(tokens), id_to_word(tokens)
    X, y = generate_training_data(tokens, word_to_id)
