import torch
import numpy as np
import pickle
from gensim.models import Word2Vec
from torch.utils.data import DataLoader
from src.model import TripMindEncoder, TripMindTrainer
from src.dataset import TripMindDataset
from sklearn.preprocessing import LabelEncoder
import json
import torch.nn as nn

# Configurations
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
W2V_PATH = 'weights/tripmind_w2v.model'
DATA_PATH = 'data/cleaned_data.jsonl'
EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM = 128, 128, 128

EPOCHS = 50
BATCH_SIZE = 32
LR = 0.001
NUM_CLASSES = len(le.classes_)

w2v_model = Word2Vec.load(W2V_PATH)
word2idx = {word: i + 2 for i, word in enumerate(w2v_model.wv.index_to_key)}
word2idx['<PAD>'], word2idx['<UNK>'] = 0, 1

embed_matrix = np.zeros((len(word2idx), EMBED_DIM))
for word, i in word2idx.items():
    if word in w2v_model.wv: embed_matrix[i] = w2v_model.wv[word]

# Prepare Label Encoder
cats = []
with open(DATA_PATH, 'r') as f:
    for l in f:
        c = json.loads(l).get("categories")
        try:
            name = (json.loads(c) if isinstance(c, str) else c)[0]['name']
            cats.append(name)
        except: pass
le = LabelEncoder().fit(cats)

# Training Setup
encoder = TripMindEncoder(len(word2idx), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, embed_matrix)
trainer = TripMindTrainer(encoder, OUTPUT_DIM, len(le.classes_)).to(DEVICE)
dataset = TripMindDataset(DATA_PATH, word2idx, le)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def train_model(model, train_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Bắt đầu huấn luyện trên thiết bị: {device}")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Acc: {accuracy:.2f}%")

# Training Loops
train_model(trainer, train_loader, EPOCHS, LR, DEVICE)

# Save Model and Assets
import os
if not os.path.exists('weights'):
    os.makedirs('weights')

# Save weight of Encoder (Semantic Vector)
torch.save(encoder.state_dict(), "weights/encoder_weights.pth")

# Save all weights of Trainer (for finetune)
torch.save(trainer.state_dict(), "/weights/full_trainer_weights.pth")

# Save Label Encoder and Word2Idx for Inference
with open("weights/assets.pkl", "wb") as f:
    pickle.dump({
        'word2idx': word2idx, 
        'label_encoder': le,
        'vocab_size': len(word2idx),
        'num_classes': NUM_CLASSES
    }, f)

print("Save model!")

# After training
torch.save(encoder.state_dict(), "weights/encoder_weights.pth")
with open("weights/assets.pkl", "wb") as f:
    pickle.dump({'word2idx': word2idx, 'label_encoder': le}, f)