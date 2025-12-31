import torch
from underthesea import word_tokenize

def get_semantic_vector(text, model, assets, device, max_len=100):
    """
    Trích xuất vector ngữ nghĩa từ văn bản.
    assets: dict chứa 'word2idx'
    """
    word2idx = assets['word2idx']
    model.eval()
    
    # 1. Tự động xác định index cho UNK và PAD (Xử lý lỗi KeyError)
    # Thử tìm các biến thể: '<UNK>', '<unk>', '[UNK]' hoặc mặc định là 0
    unk_idx = word2idx.get('<UNK>', word2idx.get('<unk>', word2idx.get('[UNK]', 0)))
    pad_idx = word2idx.get('<PAD>', word2idx.get('<pad>', word2idx.get('[PAD]', 0)))
    
    with torch.no_grad():
        # 2. Tokenize văn bản
        tokens = word_tokenize(text.lower(), format="text").split()
        
        # 3. Chuyển từ sang index
        indices = [word2idx.get(t, unk_idx) for t in tokens[:max_len]]
        
        # 4. Padding cho đủ độ dài max_len
        padding_size = max_len - len(indices)
        if padding_size > 0:
            indices += [pad_idx] * padding_size
        
        # 5. Chuyển thành Tensor và đẩy lên device (mps/cpu)
        input_tensor = torch.LongTensor([indices]).to(device)
        
        # 6. Dự báo qua model
        vector = model(input_tensor)
        
        # Chuyển kết quả về list để lưu vào ChromaDB
        return vector.cpu().numpy()[0].tolist()

def preprocess_text(text):
    return word_tokenize(text.lower(), format="text").split()