import torch

# Given a single 5-letter word, produces its encoding consisting of individual one-hot letter encodings,
# as a [130] vector where the first 26 entries store the one-hot encoding for the first
# letter, the next 26 represent the one-hot encoding for the second letter, and so on.
def word_to_encoding(word):
    onehot = torch.zeros(5, 26)
    for i, c in enumerate(word):
        onehot[i, ord(c) - ord('a')] = 1.0
    return onehot.flatten()
