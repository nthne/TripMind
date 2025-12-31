from collections import Counter

def tokenize(text):
    return text.split()

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_len=100):
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens][:max_len]
    return ids + [vocab["<PAD>"]] * (max_len - len(ids))
