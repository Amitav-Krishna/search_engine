#!/usr/bin/env python
# coding: utf-8

# In[71]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import re
seaborn.set_theme()


# In[72]:


def create_id_to_word(tokens: list[str]) -> dict[int, str]:
    return {i: word for i, word in enumerate(sorted(set(tokens)))}


# In[73]:


def tokenize(text, lowercase=True, keep_apostrophes=True, split_hyphens=False):
    if lowercase:
        text = text.lower()

    if keep_apostrophes:
        text = re.sub(r"'\w\b", "", text)

    tokens = []
    if split_hyphens:
        tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
    else:
        tokens = re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", text)

    tokens = [t for t in tokens if t and t != "'"]

    return tokens


# In[74]:


def create_word_to_id(tokens: list[str]) -> dict[str, int]:
    return {word: i for i, word in enumerate(sorted(set(tokens)))}


# In[75]:


text = '''The team at Google’s DeepMind are one of the impactful groups in the field of AI today. From beating the world champion of Go (A very hard Chinese board game), to beating the world champion of StarCraft, this team is truly ready to overcome any challenge. While beating up nerds is a very honourable cause, DeepMind has also been tackling another big challenge: protein-folding. Since 1994, every 2 years, the Critical Assessment of protein Structure Prediction (CASP) has been dedicated to reducing the time, effort, and money needed to predict the structure of proteins. They release around 100 amino acid sequences whose structure has been found in the lab, but not revealed to the public. These teams then attempt to find the structure of these proteins and are graded using a spherical distance test (GDT), which measures on a scale from 0 to 100 how close a predicted structure is from the shape of a protein identified in the lab. In 2020, DeepMind shocked the world by managing to predict the structures of over two thirds of the proteins with a score of above 90, or within the width of an atom. For any score above 90, the differences between the predicted and actual structure could be down to problems with the experiments. Proteins are also fundamentally floppy, so a score above 90 could be within the range of natural variation.'''

# Tokenize the text
tokens = tokenize(text)

# Create word to id mappings
word_to_id = create_word_to_id(tokens)
id_to_word = create_id_to_word(tokens)


# In[76]:


def one_hot_encode(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res


# In[77]:


def human_readable_vectors(centre_vec: list[int], context_vec: list[int],
                         id_to_word_map: dict[int, str]) -> str:
    centre_word = id_to_word_map[np.argmax(centre_vec)]
    context_word = id_to_word_map[np.argmax(context_vec)]
    return f"{centre_word} → {context_word}"


# In[78]:


def generate_training_data(tokens, word_to_id, window=2):
    X = [] # stores one_hot_encoded centre words
    y = [] # stores one hot encoded context words
    n_tokens = len(tokens) # the number of tokens

    id_to_word_map = create_id_to_word(tokens)

    for i in range(n_tokens):
        left = list(range(max(0, i - window), i)) # A range of numbers to the specified distance to the left
        right = list(range(i+1, min(n_tokens, i + window + 1))) # A range of numbers to the specified distance to the right

        for j in left + right: # Loops through the tokens to the right or left (context tokens)
            if i == j:
                continue
            X.append(one_hot_encode(word_to_id[tokens[i]], len(word_to_id)))
            y.append(one_hot_encode(word_to_id[tokens[j]], len(word_to_id)))
            #print(human_readable_vectors(X[-1], y[-1], id_to_word_map))

    return np.array(X), np.array(y)


# In[ ]:


def init_network(vocab_size, n_embedding):
    model = {
        "w1": np.random.randn(vocab_size, n_embedding),
        "w2": np.random.randn(n_embedding, vocab_size)
    }
    return model


# In[79]:


# Generate training data
X, y = generate_training_data(tokens, word_to_id, window=2)

# Get vocabulary size
vocab_size = len(word_to_id)


# In[80]:


def softmax(x, epsilon=1e-10):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
    return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + epsilon)


# In[81]:


def cross_entropy(z, y):
    z = np.clip(z, 1e-10, 1-1e-10)  # Avoid log(0)
    return -np.sum(y * np.log(z))


# In[82]:


def backward(model, X, y, alpha, clip_value=5.0):
    cache = forward(model, X)

    # Gradient calculations
    da2 = cache["z"] - y
    dw2 = cache["a1"].T @ da2
    da1 = da2 @ model["w2"].T
    dw1 = X.T @ da1

    # Gradient clipping
    dw1 = np.clip(dw1, -clip_value, clip_value)
    dw2 = np.clip(dw2, -clip_value, clip_value)

    # Update weights
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2

    return cross_entropy(cache["z"], y)




# In[83]:


def forward(model, X, return_cache=True):
    cache = {}
    cache["a1"] = X @ model["w1"]
    cache["a2"] = cache["a1"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])
    return cache if return_cache else cache["z"]


# In[84]:


n_embedding = 10
model = init_network(vocab_size, n_embedding)

# Training parameters
n_iter = 50
learning_rate = 0.05

history = []
for i in range(n_iter):
    loss = backward(model, X, y, learning_rate)
    history.append(loss)
    if i % 10 == 0:
        print(f"Iteration {i}: Loss = {loss:.4f}")


# In[85]:


plt.figure(figsize=(8, 4))
plt.plot(range(len(history)), history, color="skyblue")
plt.title("Training Loss Over Iterations", pad=20)
plt.xlabel("Iteration")
plt.ylabel("Cross Entropy Loss")
plt.grid(True, alpha=0.3)
plt.show()


# In[88]:


learning = one_hot_encode(word_to_id["deepmind"], len(word_to_id))
result = forward(model, [learning], return_cache=False)[0]

for word in (id_to_word[id] for id in np.argsort(result)[::-1]):
    print(word)
