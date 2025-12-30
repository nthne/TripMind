import torch
import pickle
import os
import json
from src.model import TripMindEncoder
from src.database import agent_1_output

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & THIẾT BỊ
# ==========================================
# Trên Kaggle, trọng số thường nằm trong /kaggle/input/ hoặc /kaggle/working/
WEIGHTS_PATH = "/Users/trannguyenmyanh/Documents/TripMind/agent/weights/" 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. HÀM LOAD MODEL & TÀI SẢN (ASSETS)
# ==========================================
def load_system(path=WEIGHTS_PATH):
    print(f"--- Đang khởi tạo hệ thống trên {DEVICE} ---")
    
    # a. Load assets (word2idx, label_encoder, vocab_size)
    asset_file = os.path.join(path, "assets.pkl")
    with open(asset_file, "rb") as f:
        assets = pickle.load(f)
    
    word2idx = assets['word2idx']
    vocab_size = assets['vocab_size']
    
    # b. Khởi tạo cấu trúc Encoder (phải khớp tham số 128 với lúc train)
    model = TripMindEncoder(
        vocab_size=vocab_size, 
        embedding_dim=128, 
        hidden_dim=128, 
        output_dim=128
    )
    
    # c. Nạp trọng số đã huấn luyện thành công
    model_file = os.path.join(path, "encoder_weights.pth")
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Chuyển sang chế độ dự đoán
    
    print("--- Hệ thống Agent 1 đã sẵn sàng! ---")
    return model, word2idx

# ==========================================
# 3. CHƯƠNG TRÌNH CHẠY THỬ (MAIN)
# ==========================================
if __name__ == "__main__":
    # Load model và từ điển
    try:
        trained_model, w2idx = load_system()
        
        # Giả lập input từ người dùng
        # Trong thực tế, các thông tin này sẽ được Agent điều phối lấy từ chat
        user_query = "Tôi muốn tìm một khu rừng yên bình để ngắm chim và đi thuyền"
        target_province = "311303" # Ví dụ An Giang
        target_trip_type = "family"  # Lọc theo nhóm gia đình
        
        print(f"\nCâu hỏi: '{user_query}'")
        print(f"Đang tìm kiếm tại Province ID: {target_province} cho đối tượng: {target_trip_type}...")

        # Gọi Agent 1 để truy xuất 10 địa điểm phù hợp nhất
        results = agent_1_output(
            user_query=user_query,
            model=trained_model,
            word2idx=w2idx,
            device=DEVICE,
            province_id=target_province,
            trip_type=target_trip_type
        )

        # Hiển thị kết quả
        if not results:
            print("Không tìm thấy địa điểm nào phù hợp với yêu cầu của bạn.")
        else:
            print(f"\nTOP 10 ĐỊA ĐIỂM GỢI Ý CHO BẠN:")
            print("-" * 50)
            for i, loc in enumerate(results):
                print(f"{i+1}. {loc['name'].upper()}")
                print(f"   - Loại hình: {loc['category']}")
                print(f"   - Đánh giá: {loc['rating']}*")
                print(f"   - Review tiêu biểu: {loc['matching_review'][:]}...")
                print(f"   - Độ khớp (Score): {loc['score']:.4f}")
                print("-" * 50)

    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file trọng số trong thư mục weights/. Hãy đảm bảo bạn đã chạy train_pipeline.py trước.")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")