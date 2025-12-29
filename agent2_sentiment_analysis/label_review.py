import torch
from src.model import LabelReviewModel
from src.utils import encode
from src.data_preprocessing import clean_text

checkpoint = torch.load("agent2_sentiment_analysis/sentiment_checkpoint.pt", map_location="cpu")

vocab = checkpoint["vocab"]

model = LabelReviewModel(len(vocab))
model.load_state_dict(checkpoint["model_state"])

def predict_proba(text):
    model.eval()
    x = torch.tensor([encode(clean_text(text), vocab)])
    with torch.no_grad():
        p = model(x).item()
    # print(text, p)
    return p

# Input: Danh sách các địa điểm và các review của nó
# Output: Danh sách các địa điểm với ranking từ tốt nhất đến thấp nhất (Chỉ trả về name và score)
def ranking_place(list_places, threshold = 0.6):

    places_with_score = []

    for place in list_places:
        place_with_score = {"name": place["name"]}
        positive_ratio = 0
        sentiment_score = 0

        for cur_rv in place["reviews"]:
            p_rv = predict_proba(cur_rv)
            if p_rv >= threshold:
                positive_ratio += 1
            sentiment_score += p_rv
        
        positive_ratio /= len(place["reviews"])
        sentiment_score /= len(place["reviews"])

        place_with_score["final_score"] = 0.7 * sentiment_score + 0.3 * positive_ratio

    sorted_places = sorted(places_with_score, key=lambda x: x["final_score"], reverse=True)

    return(sorted_places)
 
predict_proba("Thực tế thì giá vé khá cao so với những gì mình nhận được theo ý kiến riêng. 100k/người, svien thì 50k, trẻ em dưới 16t thì free. Tuy vậy khu vực tham quan ko có nhiều, các chỉ dẫn khá thưa thớt, các biển bảng ghi thông tin cũng ko đc chăm chút chỉnh chu lắm. Đổi lại thì khu vực nhà trưng bày là 1 điểm rất sáng giá, các hiệu ứng hình ảnh thể hiện tốt các hoa văn xưa cũ, rất có tính mỹ học, nên có thêm thông tin về các kĩ thuật sử dụng, hoặc các thông tin về di vật chi tiết hơn như di tích nhà tù Hỏa Lò chẳng hạn, điều này là có thể làm được với 1 di tích Hoàng Thành Thăng Long có lịch sử lâu đời của riêng nó")