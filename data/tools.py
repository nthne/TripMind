import pandas as pd

def process_data(input_path, output_path):
    # 1. Đọc file JSONL
    df = pd.read_json(input_path, lines=True)
    
    # 2. Thêm cột id_review ở đầu (4 chữ số, từ 0000)
    # Chúng ta tạo một list các chuỗi định dạng %04d
    id_review_list = [f"{i:04d}" for i in range(len(df))]
    df.insert(0, 'review_id', id_review_list)
    
    # 3. Đánh lại số province_id theo alphabet của new_province_x
    # Lấy danh sách các tỉnh duy nhất và sắp xếp alphabet
    unique_provinces = sorted(df['new_province_x'].unique().astype(str))
    
    # Tạo dictionary để map: {'An Giang': '00', 'Bà Rịa - Vũng Tàu': '01', ...}
    province_mapping = {name: f"{i:02d}" for i, name in enumerate(unique_provinces)}
    
    # Áp dụng mapping vào cột province_id
    df['province_id'] = df['new_province_x'].map(province_mapping)
    
    # 4. Lưu lại file JSONL mới
    df.to_json(output_path, orient='records', lines=True, force_ascii=False)
    
    print(f"--- Hoàn thành ---")
    print(f"- Đã thêm {len(df)} ID review từ 0000.")
    print(f"- Đã đánh lại {len(unique_provinces)} province_id từ 00 đến {len(unique_provinces)-1:02d}.")
    print(f"- File lưu tại: {output_path}")

# Sử dụng
process_data('/Users/trannguyenmyanh/Documents/TripMind/data/final_data_with_correct_names.jsonl', '/Users/trannguyenmyanh/Documents/TripMind/data/cleaned_data.jsonl')