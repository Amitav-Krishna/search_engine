import re

def tokenization(string):
    tokens = re.findall(r"\w+|'\w+|[^\w\s]", string)
    return tokens
