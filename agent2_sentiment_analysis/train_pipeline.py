from torch.utils.data import DataLoader
import torch

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y in dataloader:
        optimizer.zero_grad()

        preds = model(x)
        loss = criterion(preds, y.float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate_loss(model, dataloader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            preds = model(x)
            loss = criterion(preds, y.float())
            total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            preds = model(x)
            pred_labels = (preds > 0.5).long()

            correct += (pred_labels == y).sum().item()
            total += len(y)

    return correct / total


from sklearn.metrics import f1_score
import numpy as np
import torch

def evaluate_f1(model, dataloader, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            probs = model(x)
            preds = (probs > threshold).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return f1_score(all_labels, all_preds)

import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
from src.data_preprocessing import split_raw_data, clean_text
from src.utils import build_vocab
from src.dataset import ReviewDataset
from src.model import LabelReviewModel
texts, labels, neutral = split_raw_data("data/cleaned_data.jsonl")

vocab = build_vocab(texts)

X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels,
    test_size=0.3,      # 70% train, 30% còn lại
    random_state=42,
    stratify=labels     # QUAN TRỌNG cho sentiment
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,      # 15% val, 15% test
    random_state=42,
    stratify=y_temp
)

train_ds = ReviewDataset(X_train, y_train, vocab)
val_ds   = ReviewDataset(X_val, y_val, vocab)
test_ds  = ReviewDataset(X_test, y_test, vocab)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)
test_loader  = DataLoader(test_ds, batch_size=32)

model = LabelReviewModel(len(vocab))
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

from src.early_stopping import EarlyStopping
early_stop = EarlyStopping(patience=3)

for epoch in range(50):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss   = evaluate_loss(model, val_loader, criterion)
    val_acc = evaluate(model, val_loader)
    f1 = evaluate_f1(model, val_loader, threshold=0.5)

    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, Validation F1={f1:.4f}")

    if early_stop.step(val_loss):
        print("⏹ Early stopping triggered")
        break
    
best_f1, best_t = 0, 0.5

for t in np.arange(0.1, 0.9, 0.05):
    f1 = evaluate_f1(model, val_loader, threshold=t)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print("Best threshold:", best_t)
print("Best F1:", best_f1)

test_acc = evaluate(model, test_loader)
f1 = evaluate_f1(model, test_loader, threshold=best_t)
print(f"Final Test Accuracy: {test_acc:.4f}, Final F1 - Score: {f1:.4f}")

# Đánh nhãn các review 2 - 4* bằng model đã train ở trên -> Lấy thêm data -> Train tiếp
print("Use above model to evaluate and label neutral review --> Add new labeled sample to dataset to train continuously")

from src.utils import encode

def predict_proba(text):
    model.eval()
    x = torch.tensor([encode(clean_text(text), vocab)])
    with torch.no_grad():
        p = model(x).item()
    # print(text, p)
    return p

print(predict_proba("Đi thời điểm này trung tuần tháng 8 ít thấy chim, không nhiều bằng rừng Tràm Tràm Sư, dịch vụ nghèo nàn"))
pseudo_pos = []
pseudo_neg = []

for d in neutral:
    p = predict_proba(d["text"])

    # Chỉ những sample chắc chắn mới label
    if p > 0.8:
        pseudo_pos.append((d["text"], 1))
    elif p < 0.2:
        pseudo_neg.append((d["text"], 0))

print(len(pseudo_pos), len(pseudo_neg))

aug_texts = texts + [x[0] for x in pseudo_pos + pseudo_neg]

aug_labels = labels + [x[1] for x in pseudo_pos + pseudo_neg]

X_train, X_temp, y_train, y_temp = train_test_split(
    aug_texts, aug_labels,
    test_size=0.3,      # 70% train, 30% còn lại
    random_state=42,
    stratify=aug_labels     # QUAN TRỌNG cho sentiment
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,      # 15% val, 15% test
    random_state=42,
    stratify=y_temp
)

train_ds = ReviewDataset(X_train, y_train, vocab)
val_ds   = ReviewDataset(X_val, y_val, vocab)
test_ds  = ReviewDataset(X_test, y_test, vocab)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)
test_loader  = DataLoader(test_ds, batch_size=32)

early_stop = EarlyStopping(patience=2)

for epoch in range(25):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss   = evaluate_loss(model, val_loader, criterion)
    val_acc = evaluate(model, val_loader)
    f1 = evaluate_f1(model, val_loader, threshold=best_t)

    print(f"[Self-Train] Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, Validation F1={f1:.4f}")

    if early_stop.step(val_loss):
        print("⏹ Early stopping triggered")
        break

# tuning threshole for f1 score
best_f1, best_t = 0, 0.5

for t in np.arange(0.1, 0.9, 0.05):
    f1 = evaluate_f1(model, val_loader, threshold=t)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print("Best threshold:", best_t)
print("Best F1:", best_f1)

test_acc = evaluate(model, test_loader)
f1 = evaluate_f1(model, test_loader, threshold=best_t)
print(f"Final Test Accuracy: {test_acc:.4f}, Final F1 - Score: {f1:.4f}")

# Test thử
print(predict_proba("Thực tế thì giá vé khá cao so với những gì mình nhận được theo ý kiến riêng. 100k/người, svien thì 50k, trẻ em dưới 16t thì free. Tuy vậy khu vực tham quan ko có nhiều, các chỉ dẫn khá thưa thớt, các biển bảng ghi thông tin cũng ko đc chăm chút chỉnh chu lắm. Đổi lại thì khu vực nhà trưng bày là 1 điểm rất sáng giá, các hiệu ứng hình ảnh thể hiện tốt các hoa văn xưa cũ, rất có tính mỹ học, nên có thêm thông tin về các kĩ thuật sử dụng, hoặc các thông tin về di vật chi tiết hơn như di tích nhà tù Hỏa Lò chẳng hạn, điều này là có thể làm được với 1 di tích Hoàng Thành Thăng Long có lịch sử lâu đời của riêng nó"))

# Save model
checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "vocab": vocab,
    "epoch": epoch
}

torch.save(checkpoint, "agent2_sentiment_analysis/sentiment_checkpoint.pt")
