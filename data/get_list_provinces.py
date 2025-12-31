import json

data_path = "data/cleaned_data.jsonl"
result_all_provinces = "data/list_provinces.json"

provinces = []
with open(data_path, "r", encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if (data["new_province_x"], data["province_id"]) not in provinces:
            provinces.append((data["new_province_x"], data["province_id"]))

with open(result_all_provinces, "w", encoding='utf-8') as f:
    json.dump(provinces, f, ensure_ascii=False, indent=2)
