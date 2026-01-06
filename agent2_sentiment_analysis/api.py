from fastapi import FastAPI, HTTPException, Body
from typing import List, Dict, Any
import uvicorn
import logging

# Import hàm ranking_place đã được tối ưu từ file label_review.py
from label_review import ranking_place

# Cấu hình logging để dễ dàng theo dõi lỗi trên terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Agent2-Evaluator")

app = FastAPI(
    title="TripMind Agent 2 - Sentiment Ranker",
    description="API này nhận danh sách địa điểm và đánh giá dựa trên cảm xúc của review.",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    """Kiểm tra xem Agent 2 có đang sống không"""
    return {"status": "healthy", "agent": "Agent 2 - Sentiment Evaluator"}

@app.post("/ranking")
async def ranking(list_places: List[Dict[str, Any]] = Body(...)):
    """
    Endpoint nhận danh sách địa điểm từ Agent 1 và trả về top 5 địa điểm tốt nhất.
    Dữ liệu mong đợi: [{"destination_id": "...", "name": "...", "reviews": ["...", "..."]}, ...]
    """
    try:
        if not list_places:
            logger.warning("Nhận được danh sách trống từ Agent 1")
            return []

        logger.info(f"Đang xử lý đánh giá cho {len(list_places)} địa điểm...")
        
        # Gọi hàm logic xử lý sentiment và ranking
        # Hàm này đã được tối ưu Batch Prediction trong bước trước
        ranked_results = ranking_place(list_places)
        
        # Chỉ trả về Top 5 địa điểm có điểm số cao nhất
        top_5 = ranked_results[:5]
        
        logger.info(f"Hoàn thành! Đã chọn ra {len(top_5)} địa điểm tốt nhất.")
        return top_5

    except Exception as e:
        logger.error(f"Lỗi khi thực hiện Ranking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    # Agent 2 chạy trên port 8000
    print("🚀 TripMind Agent 2 đang khởi động tại http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

