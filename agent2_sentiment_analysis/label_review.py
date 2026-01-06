import torch
from src.model import LabelReviewModel
from src.utils import encode
from src.data_preprocessing import clean_text

# Khởi tạo Model và Vocab từ Checkpoint
checkpoint = torch.load("F:/HUST/Năm ba/DL/prj/TripMind/TripMind/agent2_sentiment_analysis/sentiment_checkpoint.pt", map_location="cpu")
vocab = checkpoint["vocab"]
model = LabelReviewModel(len(vocab))
model.load_state_dict(checkpoint["model_state"])
model.eval()

def predict_proba(text):
    model.eval()
    x = torch.tensor([encode(clean_text(text), vocab)])
    with torch.no_grad():
        p = model(x).item()
    print(text, p)
    return p

def predict_batch(texts):
    """Dự đoán cảm xúc cho một danh sách văn bản cùng lúc (Batch Prediction)"""
    print(texts)
    if not texts:
        return []
    
    # 1. Tiền xử lý và mã hóa toàn bộ danh sách
    encoded_batch = [encode(clean_text(t), vocab) for t in texts]
    x = torch.tensor(encoded_batch)
    
    # 2. Chạy qua mô hình
    with torch.no_grad():
        probs = model(x)
        print(probs.item())
        # Xử lý trường hợp mô hình squeeze mất chiều batch khi chỉ có 1 review
        if probs.dim() == 0:
            return [probs.item()]
        return probs.tolist()

def ranking_place(list_places, threshold=0.6):
    """
    Agent 2: Xếp hạng các địa điểm dựa trên phân tích cảm xúc.
    - Input: List địa điểm từ Agent 1 (mỗi địa điểm có mảng reviews)
    - Output: Top địa điểm đã được chấm điểm và sắp xếp
    """
    if not list_places:
        return []

    all_reviews_text = []
    place_map = [] # Lưu vết review này thuộc về địa điểm thứ mấy trong list_places
    
    # BƯỚC 1: Gom tất cả review của tất cả địa điểm vào một list lớn
    for i, place in enumerate(list_places):
        reviews = place.get("reviews", [])
        for rv in reviews:
            all_reviews_text.append(rv)
            place_map.append(i)
            
    # Nếu không có review nào để đánh giá, trả về danh sách với score = 0
    if not all_reviews_text:
        return sorted([
            {
                "destination_id": p.get("destination_id"),
                "name": p.get("name"),
                "final_score": 0.0,
                "reviews": p.get("reviews", [])[:3]
            } for p in list_places
        ], key=lambda x: x["final_score"], reverse=True)

    # BƯỚC 2: Dự đoán cảm xúc cho toàn bộ batch (Nhanh hơn gọi lẻ tẻ)
    all_probs = predict_batch(all_reviews_text)
    
    # BƯỚC 3: Nhóm kết quả dự đoán về lại từng địa điểm ban đầu
    results_by_place = {i: [] for i in range(len(list_places))}
    for prob, place_idx in zip(all_probs, place_map):
        results_by_place[place_idx].append(prob)
        
    # BƯỚC 4: Tính toán điểm số tổng hợp (Final Score)
    final_results = []
    for i, place in enumerate(list_places):
        scores = results_by_place[i]
        
        if not scores:
            sentiment_score = 0.0
            positive_ratio = 0.0
        else:
            # Điểm cảm xúc trung bình (0.0 -> 1.0)
            sentiment_score = sum(scores) / len(scores)
            # Tỷ lệ review vượt ngưỡng tích cực
            positive_count = sum(1 for s in scores if s >= threshold)
            positive_ratio = positive_count / len(scores)
            
        # Công thức Ranking: 70% dựa trên cảm xúc chung, 30% dựa trên tỷ lệ hài lòng
        final_score = (0.7 * sentiment_score) + (0.3 * positive_ratio)
        
        final_results.append({
            "destination_id": place.get("destination_id"),
            "name": place.get("name"),
            "final_score": round(final_score, 3),
            "reviews": place.get("reviews", [])[:3] # Giữ lại 3 review tiêu biểu
        })
    
    # BƯỚC 5: Sắp xếp từ tốt nhất đến thấp nhất
    return sorted(final_results, key=lambda x: x["final_score"], reverse=True)

# Test nhanh một trường hợp
if __name__ == "__main__":
    example_input = [
        {
            "destination_id": "123",
            "name": "Chùa Bà Chúa Xứ",
            "reviews": ["Nơi này rất linh thiêng", "Cảnh đẹp và thanh tịnh"]
        }
    ]
    # print(ranking_place(example_input))
    print(predict_batch(["Đi thời điểm này trung tuần tháng 8 ít thấy chim, không nhiều bằng rừng Tràm Tràm Sư, dịch vụ nghèo nàn"]))
    (predict_proba("Nhà thờ có kiến trúc theo kiểu nhà rông, nghĩa là có đặc biệt. Đến thì nên qua thăm. Không đặc sắc."))
