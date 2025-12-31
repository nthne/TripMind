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

# Global variables
MODEL = None
WORD2IDX = None
DEVICE = None
PROVINCE_STATS = None
AGENT_2_URL = "http://localhost:8000/ranking" # URL của Agent 2 (FastAPI)

def load_system(weights_path="/Users/trannguyenmyanh/Documents/TripMind/agent/weights"):
    global MODEL, WORD2IDX, DEVICE, PROVINCE_STATS
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"🚀 Khởi tạo trên {DEVICE}...")
    
    try:
        # 1. Load Assets
        with open(os.path.join(weights_path, "assets.pkl"), "rb") as f:
            assets = pickle.load(f)
        WORD2IDX = assets['word2idx']
        vocab_size = assets['vocab_size']
        
        # 2. Load Model
        MODEL = TripMindEncoder(vocab_size, d_model=128, nhead=8, num_layers=4)
        MODEL.load_state_dict(torch.load(os.path.join(weights_path, "encoder_weights.pth"), map_location=DEVICE))
        MODEL.to(DEVICE)
        MODEL.eval()
        
        # 3. Load Stats từ DB
        PROVINCE_STATS = get_provinces_stats()
        logger.info(f"✓ Loaded {len(PROVINCE_STATS)} provinces")
    except Exception as e:
        logger.error(f"❌ Lỗi khởi động hệ thống: {e}")
        raise e

@app.route('/api/v1/recommend', methods=['POST'])
def recommend_places():
    """Hàm duy nhất xử lý Recommend: Kết nối Agent 1 -> Agent 2"""
    try:
        data = request.get_json()
        query = data.get('query')
        province_id = data.get('province_id')
        
        if not query or province_id is None:
            return jsonify({"success": False, "error": "Missing query or province_id"}), 400
        
        p_id_str = str(province_id).zfill(2) 
        trip_type = data.get('trip_type', 'any')
        n_places = min(int(data.get('n_places', 10)), 50)
        max_reviews = int(data.get('max_reviews_per_place', 5))
        
        # BƯỚC 1: Gọi Agent 1 (Recall) - Lấy dư ra (15 cái) để Agent 2 lọc lại
        candidates = agent_1_output(
            user_query=query,
            model=MODEL,
            word2idx=WORD2IDX,
            device=DEVICE,
            province_id=p_id_str,
            trip_type=trip_type,
            n_places=15, 
            max_reviews_per_place=max_reviews
        )
        
        if not candidates:
            return jsonify({"success": True, "data": [], "message": "No candidates found"}), 200

        # BƯỚC 2: Gọi Agent 2 (Ranking) sang FastAPI
        try:
            logger.info(f"Gửi {len(candidates)} ứng viên sang Agent 2...")
            response = requests.post(AGENT_2_URL, json=candidates, timeout=5)
            
            if response.status_code == 200:
                # Lấy Top 5 (hoặc n_places) từ Agent 2
                final_results = response.json()[:n_places]
                logger.info("Agent 2 trả về kết quả thành công.")
            else:
                logger.warning(f"Agent 2 trả về lỗi {response.status_code}, dùng kết quả thô.")
                final_results = candidates[:n_places]
        except Exception as e:
            logger.error(f"Không thể kết nối Agent 2: {e}. Trả về kết quả fallback.")
            final_results = candidates[:n_places]

        return jsonify({
            "success": True,
            "data": final_results,
            "metadata": {
                "province_id": p_id_str,
                "candidates_retrieved": len(candidates),
                "final_count": len(final_results)
            }
        }), 200

    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/provinces', methods=['GET'])
def get_provinces():
    if PROVINCE_STATS:
        provinces = [{"province_id": pid, **stats} for pid, stats in sorted(PROVINCE_STATS.items(), key=lambda x: x[1]['total_reviews'], reverse=True)]
        return jsonify({"success": True, "total_provinces": len(provinces), "provinces": provinces}), 200
    return jsonify({"success": False, "error": "Stats not available"}), 500

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "device": str(DEVICE), "agent2_link": AGENT_2_URL}), 200

if __name__ == "__main__":
    load_system()
    app.run(host='0.0.0.0', port=5000, debug=False)