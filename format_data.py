import pandas as pd

def add_review_id_column(input_path, output_path):
    # 1. Đọc file JSONL
    df = pd.read_json(input_path, lines=True)
    
    # 2. Tạo dãy số ID và định dạng thành 4 chữ số (ví dụ: 0000, 0001,...)
    # zfill(4) giúp tự động thêm các số 0 để đủ độ dài là 4
    new_ids = [str(i).zfill(4) for i in range(len(df))]
    
    # 3. Chèn cột mới vào vị trí đầu tiên (index = 0)
    df.insert(0, 'id_review', new_ids)
    
    # 4. Lưu lại file JSONL mới
    df.to_json(output_path, orient='records', lines=True, force_ascii=False)
    
    print(f"Hoàn thành! Đã thêm cột 'id_review' từ {new_ids[0]} đến {new_ids[-1]}")
    print(f"File lưu tại: {output_path}")

# Thực thi
add_review_id_column('/Users/trannguyenmyanh/Documents/TripMind/data/cleaned_data.jsonl',
                      '/Users/trannguyenmyanh/Documents/TripMind/data/cleaned_data_final.jsonl')