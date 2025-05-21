def word_to_id(tokens: list[str]) -> dict[str, int]:
    return {word: i for i, word in enumerate(sorted(set(tokens)))}

def id_to_word(tokens: list[str]) -> dict[int, str]:
    return {i: word for i, word in enumerate(sorted(set(tokens)))}
