import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

import json

def load_raw_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def split_raw_data(path):
    data = load_raw_data(path)

    pos = [d for d in data if d["rating_x"] == 5]
    neg = [d for d in data if d["rating_x"] in [1, 2]]
    neutral = [d for d in data if d["rating_x"] not in [1, 2, 5]]

    neutral += pos[len(neg):]
    pos = pos[:len(neg)]
    print(len(pos), len(neg), len(neutral))

    train_texts = [clean_text(d["text"]) for d in pos + neg]
    train_labels = [1] * len(pos) + [0] * len(neg)

    return train_texts, train_labels, neutral





