import torch.nn as nn

class LabelReviewModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)              
        out, _ = self.lstm(emb)              

        pooled = out.mean(dim=1)            
        logits = self.fc(pooled)            

        return self.sigmoid(logits).squeeze()
