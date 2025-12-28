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

sentences = load_and_tokenize('/kaggle/input/dl-data/cleaned_data.jsonl')

# Cấu hình các tham số quan trọng
EMBEDDING_DIM = 128
model_w2v = Word2Vec(
    sentences=sentences,
    vector_size=EMBEDDING_DIM,
    window=5,      # Khoảng cách giữa từ hiện tại và từ dự đoán
    min_count=2,   # Loại bỏ các từ xuất hiện ít hơn 2 lần
    workers=1,     # Số luồng xử lý
    sg=1           # Sử dụng Skip-gram (thường tốt hơn cho tập dữ liệu nhỏ/vừa)
)

# Lưu model để dùng lại
model_w2v.save("/kaggle/working/")
