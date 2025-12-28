import torch
import torch.nn as nn

class TripMindEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_matrix=None):
        super(TripMindEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = True
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, 
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, text_indices):
        embedded = self.dropout(self.embedding(text_indices))
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.mean(lstm_out, dim=1) 
        return self.fc(pooled)

class TripMindTrainer(nn.Module):
    def __init__(self, encoder, output_dim, num_classes):
        super(TripMindTrainer, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(output_dim, num_classes)
        
    def forward(self, x):
        vector = self.encoder(x)
        return self.classifier(vector)