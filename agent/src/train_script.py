import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import TripMindEncoder
from dataset import TripMindDataset

# Khởi tạo danh sách lưu lịch sử
history = {
    'train_loss': [],
    'train_acc': []
}

# --- CẤU HÌNH ---
DATA_PATH = "/kaggle/input/dl-dataset/cleaned_data.jsonl"
WEIGHTS_DIR = "/kaggle/working/"
MAX_SEQ_LEN = 100
BATCH_SIZE = 32
EPOCHS = 100  # Tăng lên để thấy sự hội tụ
LEARNING_RATE = 5e-5
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4 
best_acc = 0.0  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TIỀN XỬ LÝ VĂN BẢN ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()


def train():
    global best_acc
    print(f"🚀 Khởi tạo Multi-task Learning trên {device}...")
    
    dataset = TripMindDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    vocab_size = len(dataset.word2idx)
    num_destinations = len(dataset.label_encoder.classes_)
    num_categories = len(dataset.cat_encoder.classes_)
    
    print(f"📊 Vocab: {vocab_size} | Địa danh: {num_destinations} | Loại hình: {num_categories}")

    # Khởi tạo Model với 2 đầu ra
    model = TripMindEncoder(
        vocab_size=vocab_size, 
        num_categories=num_categories, # Truyền vào để kích hoạt classifier
        d_model=D_MODEL, 
        nhead=NHEAD, 
        num_layers=NUM_LAYERS
    ).to(device)
    
    # Classifier phụ cho Destination ID
    dest_classifier = nn.Linear(D_MODEL, num_destinations).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(list(model.parameters()) + list(dest_classifier.parameters()), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for texts, labels_dist, labels_cat in loop:
            texts, labels_dist, labels_cat = texts.to(device), labels_dist.to(device), labels_cat.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: Model trả về Embedding và Logits của Category
            emb, cat_logits = model(texts)
            dest_outputs = dest_classifier(emb)
            
            # Tính toán 2 loại Loss
            loss_dist = criterion(dest_outputs, labels_dist) # Phân biệt địa danh
            loss_cat = criterion(cat_logits, labels_cat)    # Phân biệt loại hình (Chùa vs Biển)
            
            # Loss tổng hợp (Ưu tiên học ngữ nghĩa Category với trọng số 2.0)
            batch_loss = loss_dist + (2.0 * loss_cat)
            
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            
            # Tính accuracy dựa trên Destination (để so sánh với bản cũ)
            _, predicted = torch.max(dest_outputs.data, 1)
            total += labels_dist.size(0)
            correct += (predicted == labels_dist).sum().item()
            
            epoch_acc = 100 * correct / total
            loop.set_postfix(loss=f"{total_loss/len(dataloader):.4f}", acc=f"{epoch_acc:.2f}%")

        # Lưu lịch sử để vẽ biểu đồ
        history['train_loss'].append(total_loss/len(dataloader))
        history['train_acc'].append(epoch_acc)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "encoder_weights.pth"))
            print(f"🌟 Best Model Updated: {best_acc:.2f}%")

    # Lưu Assets bao gồm cả cat_encoder
    assets = {
        "word2idx": dataset.word2idx,
        "vocab_size": vocab_size,
        "label_encoder": dataset.label_encoder,
        "cat_encoder": dataset.cat_encoder,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS
    }
    with open(os.path.join(WEIGHTS_DIR, "assets.pkl"), "wb") as f:
        pickle.dump(assets, f)

if __name__ == "__main__":
    train()