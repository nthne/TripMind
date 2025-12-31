from flask import Flask, request, jsonify
import requests
import torch
import pickle
import os
import logging
from model import TripMindEncoder
from database import get_provinces_stats, agent_1_output

# Cấu hình Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- GLOBAL VARIABLES & CONFIG ---
MODEL = None
WORD2IDX = None
ASSETS = None
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
PROVINCE_STATS = None

# URLs của các Agent thành viên
AGENT_2_URL = "http://localhost:8000/ranking"
AGENT_3_URL = "http://localhost:9000/optimize"

def load_system():
    global MODEL, ASSETS, WORD2IDX, PROVINCE_STATS
    try:
        logger.info(f"🚀 Khởi tạo hệ thống trên thiết bị: {DEVICE}...")
        weights_path = "/Users/trannguyenmyanh/Documents/TripMind/agent/weights"
        
        # 1. Load Assets
        with open(os.path.join(weights_path, "assets.pkl"), "rb") as f:
            ASSETS = pickle.load(f)
        
        WORD2IDX = ASSETS['word2idx']
        vocab_size = ASSETS['vocab_size']
        num_categories = len(ASSETS['cat_encoder'].classes_)
        
        # 2. Khởi tạo Model Multi-task Transformer
        MODEL = TripMindEncoder(
            vocab_size=vocab_size,
            num_categories=num_categories,
            d_model=256,   
            nhead=8,
            num_layers=4   
        ).to(DEVICE)
        
        # 3. Load trọng số
        weights_file = os.path.join(weights_path, "encoder_weights.pth")
        state_dict = torch.load(weights_file, map_location=DEVICE)
        MODEL.load_state_dict(state_dict)
        MODEL.eval()
        
        # 4. Load thống kê tỉnh thành
        PROVINCE_STATS = get_provinces_stats()
        
        logger.info("✅ Chuỗi 3 Agent đã sẵn sàng điều phối!")
        
    except Exception as e:
        logger.error(f"❌ Lỗi khởi động hệ thống: {str(e)}")
        raise e

@app.route('/api/v1/recommend', methods=['POST'])
def recommend_places():
    """Hàm trung tâm điều phối chuỗi 3 Agent"""
    try:
        data = request.get_json()
        query = data.get('query')
        province_id = data.get('province_id')
        
        if not query or province_id is None:
            return jsonify({"success": False, "error": "Missing query or province_id"}), 400
        
        p_id_str = str(province_id).zfill(2) 
        trip_type = data.get('trip_type', 'any')
        n_places = min(int(data.get('n_places', 5)), 10) # Agent 3 tối ưu tốt nhất cho 5-10 điểm
        
        logger.info(f"🔍 Nhận Query: '{query}' | Tỉnh: {p_id_str}")

        # --- BƯỚC 1: AGENT 1 (Recall) ---
        # Lấy 15 ứng viên để Agent 2 có dữ liệu để lọc
        candidates = agent_1_output(
            user_query=query,
            model=MODEL,
            word2idx=WORD2IDX,
            assets=ASSETS,
            device=DEVICE,
            province_id=p_id_str,
            trip_type=trip_type,
            n_places=15, 
            max_reviews_per_place=5
        )
        
        if not candidates:
            return jsonify({"success": True, "data": [], "message": "Không tìm thấy kết quả"}), 200

        # --- BƯỚC 2: AGENT 2 (Sentiment Ranking) ---
        try:
            logger.info("📡 Đang gửi dữ liệu sang Agent 2 (Ranking)...")
            res2 = requests.post(AGENT_2_URL, json=candidates, timeout=10)
            if res2.status_code == 200:
                ranked_places = res2.json() # Agent 2 trả về danh sách đã chấm điểm
            else:
                ranked_places = candidates
        except Exception as e:
            logger.error(f"⚠️ Lỗi kết nối Agent 2: {e}")
            ranked_places = candidates

        # Lấy Top N để đưa vào Agent 3 tối ưu lộ trình
        top_candidates = ranked_places[:n_places]

        # --- BƯỚC 3: AGENT 3 (Route Optimization) ---
        try:
            logger.info("📡 Đang gửi dữ liệu sang Agent 3 (Optimization)...")
            res3 = requests.post(AGENT_3_URL, json=top_candidates, timeout=10)
            if res3.status_code == 200:
                final_itinerary = res3.json()
            else:
                final_itinerary = top_candidates
        except Exception as e:
            logger.error(f"⚠️ Lỗi kết nối Agent 3: {e}")
            final_itinerary = top_candidates

        return jsonify({
            "success": True,
            "data": final_itinerary,
            "metadata": {
                "province_id": p_id_str,
                "candidates_retrieved": len(candidates),
                "optimization": "Full Pipeline (Recall -> Rank -> Route)"
            }
        }), 200
    
    except Exception as e:
        logger.error(f"💥 Critical Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/provinces', methods=['GET'])
def get_provinces():
    if PROVINCE_STATS:
        provinces = [{"province_id": pid, **stats} for pid, stats in sorted(PROVINCE_STATS.items(), key=lambda x: x[1]['total_reviews'], reverse=True)]
        return jsonify({"success": True, "total_provinces": len(provinces), "provinces": provinces}), 200
    return jsonify({"success": False, "error": "Stats not available"}), 500

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "device": str(DEVICE),
        "agent2": AGENT_2_URL,
        "agent3": AGENT_3_URL
    }), 200

if __name__ == "__main__":
    load_system()
    # Chạy Agent 1 trên port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)