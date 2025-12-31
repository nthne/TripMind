import requests
import time
try:
    from extract_name_from_query import extract_city_from_query
except ImportError:
    print("Cảnh báo: Không tìm thấy hàm extract_city_from_query. Vui lòng kiểm tra lại file!")
    def extract_city_from_query(q): return "00" 

# URL của Agent 1 (Cửa ngõ Gateway)
API_URL = "http://127.0.0.1:5000/api/v1/recommend"

def print_divider():
    print("-" * 60)

def test_full_pipeline(query, province_id, trip_type="any"):
    print(f" TRIPMIND MULTI-AGENT SYSTEM TEST ".center(60, "="))
    
    print(f"\nCâu lệnh người dùng: '{query}'")
    print(f"ID tỉnh xác định: {province_id}")
    print(f"Loại hình chuyến đi: {trip_type}")
    print_divider()

    if not province_id:
        print("Lỗi: Không trích xuất được ID tỉnh. Dừng kiểm tra.")
        return

    payload = {
        "query": query,
        "province_id": str(province_id),
        "trip_type": trip_type,
        "n_places": 5
    }

    try:
        print("Đang gửi yêu cầu đến Agent Gateway (Port 5000)...")
        start_time = time.time()
        
        response = requests.post(API_URL, json=payload, timeout=30)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            res_data = response.json()
            itinerary = res_data.get('data', [])
            meta = res_data.get('metadata', {})

            print(f"\nHOÀN TẤT TRONG {elapsed:.2f} GIÂY")
            print(f"Thống kê: Agent 1 đã quét {meta.get('candidates_retrieved')} ứng viên.")
            print(f"Trạng thái: {meta.get('optimization')}")
            
            print("\n" + "LỘ TRÌNH DI CHUYỂN GỢI Ý (Đã tối ưu) ".center(60, "-"))
            
            if not itinerary:
                print("∅ Không tìm thấy địa điểm phù hợp.")
            
            for i, place in enumerate(itinerary, 1):
                name = place.get('name', 'Không rõ tên')
                d_id = place.get('destination_id', 'N/A')
                sentiment_score = place.get('final_score', 'N/A')
                
                print(f" {i} ".center(5, "[").center(8, "]") + f" {name}")
                print(f"      ├─ ID: {d_id}")
                
                # Hiển thị điểm từ Agent 2
                if sentiment_score != 'N/A':
                    print(f"      ├─ Điểm chất lượng (Agent 2): {sentiment_score}")
                
                # Hiển thị review tiêu biểu
                reviews = place.get('reviews', [])
                if reviews:
                    short_review = reviews[0][:120].replace("\n", " ")
                    print(f"      └─ Đánh giá: \"{short_rev}...\"")
                print_divider()
                
            print("\nLưu ý: Thứ tự trên là lộ trình ngắn nhất do Agent 3 (DQN) tính toán.")

        else:
            print(f"Lỗi API (Status {response.status_code}):")
            print(response.text)

    except requests.exceptions.Timeout:
        print("Lỗi: Quá thời gian phản hồi (Timeout). Vui lòng kiểm tra các Agent 2 và 3.")
    except Exception as e:
        print(f"Lỗi kết nối: {e}")

if __name__ == "__main__":
    print("\n--- HỆ THỐNG TRỰC TIẾP TRIPMIND ---")
    user_query = input("Nhập yêu cầu du lịch của bạn: ")
    
    # Bước trích xuất tỉnh
    print("Đang phân tích địa danh...")
    extracted_id = extract_city_from_query(user_query)
    
    # Chạy kiểm tra
    test_full_pipeline(user_query, extracted_id, "any")