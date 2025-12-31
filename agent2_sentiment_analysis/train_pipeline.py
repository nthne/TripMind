from torch.utils.data import DataLoader
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
from src.data_preprocessing import split_raw_data, clean_text
from src.utils import build_vocab
from src.dataset import ReviewDataset
from src.model import LabelReviewModel

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

texts, labels, neutral = split_raw_data("data/final_cleaned_data_with_coords.jsonl")

vocab = build_vocab(texts)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

train_ds = ReviewDataset(X_train, y_train, vocab)
test_ds  = ReviewDataset(X_test, y_test, vocab)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32)

model = LabelReviewModel(len(vocab))
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

for epoch in range(10):
    loss = train(model, train_loader, optimizer, criterion)
    acc = evaluate(model, test_loader)
    print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")

print("Use above model to evaluate and label neutral review --> Add new labeled sample to dataset to train continuously")

from src.utils import encode

def predict_proba(text):
    model.eval()
    x = torch.tensor([encode(clean_text(text), vocab)])
    with torch.no_grad():
        p = model(x).item()
    # print(text, p)
    return p

pseudo_pos = []
pseudo_neg = []

for d in neutral:
    p = predict_proba(d["text"])

    # Chỉ những sample chắc chắn mới label
    if p > 0.8:
        pseudo_pos.append((d["text"], 1))
    elif p < 0.2:
        pseudo_neg.append((d["text"], 0))

aug_texts = texts + \
            [x[0] for x in pseudo_pos + pseudo_neg]

aug_labels = labels + \
             [x[1] for x in pseudo_pos + pseudo_neg]

X_train, X_test, y_train, y_test = train_test_split(
    aug_texts, aug_labels, test_size=0.2, random_state=42
)

train_ds = ReviewDataset(X_train, y_train, vocab)
test_ds  = ReviewDataset(X_test, y_test, vocab)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32)

for epoch in range(5):   # ít epoch hơn
    loss = train(model, train_loader, optimizer, criterion)
    acc  = evaluate(model, test_loader)
    print(f"[Self-Train] Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")

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
