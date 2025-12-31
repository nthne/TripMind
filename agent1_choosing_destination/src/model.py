import torch
import torch.nn as nn
import math

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
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:x.size(1)]

class TripMindEncoder(nn.Module):
    def __init__(self, vocab_size, num_categories=None, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        
        # enable_nested_tensor=False để tránh lỗi trên chip MPS (Mac)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers,
            enable_nested_tensor=False 
        )
        
        self.fc_emb = nn.Linear(d_model, d_model)
        self.num_categories = num_categories
        if num_categories is not None:
            self.category_classifier = nn.Linear(d_model, num_categories)

    def forward(self, x):
        padding_mask = (x == 0)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        mask = ~padding_mask.unsqueeze(-1)
        sentence_emb = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        final_embedding = self.fc_emb(sentence_emb)
        
        if self.num_categories is not None:
            cat_logits = self.category_classifier(sentence_emb)
            return final_embedding, cat_logits
            
        return final_embedding