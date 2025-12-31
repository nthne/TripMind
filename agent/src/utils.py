import torch
from underthesea import word_tokenize

# utils.py
import torch
from underthesea import word_tokenize

def get_semantic_vector(text, model, word2idx, device, max_len=100):
    model.eval()
    pad_idx = 0
    unk_idx = 1
    
    with torch.no_grad():
        # SỬA TẠI ĐÂY: Bỏ format="text" để lấy danh sách từ đơn
        tokens = word_tokenize(text.lower()) 
        
        # Chuyển từ sang index (Lúc này 'chùa' sẽ được tìm thấy trong word2idx)
        indices = [word2idx.get(t, unk_idx) for t in tokens[:max_len]]
        
        # Padding
        padding_size = max_len - len(indices)
        if padding_size > 0:
            indices += [pad_idx] * padding_size
        
        input_tensor = torch.LongTensor([indices]).to(device)
        output = model(input_tensor)
        
        if isinstance(output, tuple):
            vector = output[0]
        else:
            vector = output
            
        return vector.cpu().numpy()[0].tolist()

def preprocess_text(text):
    return word_tokenize(text.lower(), format="text").split()