def human_readable_vectors(centre_vec: list[int], context_vec: list[int],
                         id_to_word_map: dict[int, str]) -> str:
    centre_word = id_to_word_map[centre_vec.index(1)]
    context_word = id_to_word_map[context_vec.index(1)]
    return f"{centre_word} → {context_word}"
