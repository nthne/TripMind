import torch
import json
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class TripMindDataset(Dataset):
    def __init__(self, file_path, word2idx, label_encoder, max_len=100):
        self.data = []
        self.word2idx = word2idx
        self.label_encoder = label_encoder
        self.max_len = max_len
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                text = record.get("text")
                cat_raw = record.get("categories")
                try:
                    cat_list = json.loads(cat_raw) if isinstance(cat_raw, str) else cat_raw
                    cat_name = cat_list[0]['name'] if cat_list else None
                except: cat_name = None

                if text and cat_name:
                    self.data.append((text, cat_name))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        text, cat_name = self.data[idx]
        from src.utils import preprocess_text
        tokens = preprocess_text(text)
        indices = [self.word2idx.get(t, self.word2idx['<UNK>']) for t in tokens[:self.max_len]]
        indices += [self.word2idx['<PAD>']] * (self.max_len - len(indices))
        label = self.label_encoder.transform([cat_name])[0]
        return torch.LongTensor(indices), torch.tensor(label, dtype=torch.long)