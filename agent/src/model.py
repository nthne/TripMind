import torch
import torch.nn as nn
import math

class TripMindEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=2):
        super(TripMindEncoder, self).__init__()
        self.d_model = d_model
        
        # 1. Embedding Layer: Chuyển từ ID sang Vector
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Encoding: Giúp Transformer hiểu thứ tự của các từ
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder Layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. Output Layer: Nén kết quả về một vector 128 chiều đại diện cho cả câu
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [batch_size, seq_len]
        
        # Tạo mask để bỏ qua các ký tự padding (0)
        padding_mask = (x == 0) 
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer xử lý toàn bộ ngữ cảnh
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Pooling: Lấy trung bình cộng của các từ (Mean Pooling) để ra vector câu
        # Lưu ý: Chỉ lấy trung bình của các từ thực sự (không lấy padding)
        mask = ~padding_mask.unsqueeze(-1)
        sentence_emb = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        return self.fc_out(sentence_emb)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return x