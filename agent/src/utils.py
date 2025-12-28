import torch
from underthesea import word_tokenize

def get_semantic_vector(text, model, word2idx, device, max_len=100):
    model.eval()
    with torch.no_grad():
        # Sử dụng underthesea tương nhất với lúc train W2V
        tokens = word_tokenize(text.lower(), format="text").split()
        indices = [word2idx.get(t, word2idx['<UNK>']) for t in tokens[:max_len]]
        indices += [word2idx['<PAD>']] * (max_len - len(indices))
        
        input_tensor = torch.LongTensor([indices]).to(device)
        vector = model(input_tensor)
        return vector.cpu().numpy()[0].tolist()

def preprocess_text(text):
    return word_tokenize(text.lower(), format="text").split()