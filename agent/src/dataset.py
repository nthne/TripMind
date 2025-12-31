import torch
import json
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

import torch
from underthesea import word_tokenize

def get_semantic_vector(text, model, word2idx, device, max_len=100):
    model.eval()
    with torch.no_grad():
        tokens = word_tokenize(text.lower(), format="text").split()
        indices = [word2idx.get(t, word2idx['<UNK>']) for t in tokens[:max_len]]
        indices += [word2idx['<PAD>']] * (max_len - len(indices))
        
        input_tensor = torch.LongTensor([indices]).to(device)
        vector = model(input_tensor)
        return vector.cpu().numpy()[0].tolist()

def preprocess_text(text):
    return word_tokenize(text.lower(), format="text").split()

class TripMindDataset(Dataset):
    def __init__(self, data_path, word2idx=None, label_encoder=None, cat_encoder=None):
        self.texts, self.labels, self.cat_names = [], [], []
        
        # Đọc dữ liệu
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Tiền xử lý văn bản
                self.texts.append(clean_text(item['text']))
                # Lấy nhãn địa danh (Destination ID)
                self.labels.append(str(item['destination_id']))
                # Lấy nhãn loại hình (Category Name) sử dụng staticmethod của class
                cat_name = self.get_category_name(item.get('categories', '[]'))
                self.cat_names.append(cat_name)

        # 1. Xử lý Từ điển (Vocabulary)
        if word2idx is None:
            self.word2idx = {"<PAD>": 0, "<UNK>": 1}
            for tokens in self.texts:
                for token in tokens:
                    if token not in self.word2idx:
                        self.word2idx[token] = len(self.word2idx)
        else:
            self.word2idx = word2idx

        # 2. Encode nhãn Địa danh (Destination ID)
        self.label_encoder = label_encoder or LabelEncoder()
        if label_encoder is None: 
            self.y = self.label_encoder.fit_transform(self.labels)
        else: 
            self.y = self.label_encoder.transform(self.labels)

        # 3. Encode nhãn Loại hình (Category Name) - Dành cho Multi-task
        self.cat_encoder = cat_encoder or LabelEncoder()
        if cat_encoder is None: 
            self.y_cat = self.cat_encoder.fit_transform(self.cat_names)
        else: 
            self.y_cat = self.cat_encoder.transform(self.cat_names)

    @staticmethod
    def get_category_name(cat_raw):
        """Hàm bóc tách tên thể loại từ chuỗi JSON hoặc List (Đã đóng gói vào class)"""
        try:
            if isinstance(cat_raw, str):
                import json
                cat_list = json.loads(cat_raw)
            else:
                cat_list = cat_raw
                
            if isinstance(cat_list, list) and len(cat_list) > 0:
                # Lấy tên của category đầu tiên trong danh sách
                return cat_list[0].get('name', 'Khác')
        except Exception:
            pass
        return "Khác"

    def __len__(self): 
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        # Chuyển từ sang ID (mặc định UNK=1)
        indexed = [self.word2idx.get(t, 1) for t in tokens]
        
        # Thực hiện Padding hoặc Truncating cho đủ MAX_SEQ_LEN
        if len(indexed) < MAX_SEQ_LEN:
            indexed += [0] * (MAX_SEQ_LEN - len(indexed)) # PAD=0
        else:
            indexed = indexed[:MAX_SEQ_LEN]
            
        # Trả về bộ 3 dữ liệu để huấn luyện Multi-task
        return (
            torch.tensor(indexed), 
            torch.tensor(self.y[idx]), 
            torch.tensor(self.y_cat[idx])
        )