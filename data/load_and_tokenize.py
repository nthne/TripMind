import json
from underthesea import word_tokenize
from gensim.models import Word2Vec

def load_and_tokenize(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            text = record.get("text", "")
            if text:
                tokens = word_tokenize(text.lower(), format="text").split()
                sentences.append(tokens)
    return sentences

sentences = load_and_tokenize('TripMind/data/cleaned_data.jsonl')

EMBEDDING_DIM = 128
model_w2v = Word2Vec(
    sentences=sentences,
    vector_size=EMBEDDING_DIM,
    window=5,      
    min_count=2,   
    workers=1,    
    sg=1          
)

# Save model
model_w2v.save("/TripMind")
