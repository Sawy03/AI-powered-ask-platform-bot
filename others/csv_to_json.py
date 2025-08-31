import csv
import json

input_csv = "ai_generated_qa_dataset.csv"
output_jsonl = "dataset.jsonl"

with open(input_csv, newline='', encoding='utf-8') as csvfile, open(output_jsonl, 'w', encoding='utf-8') as jsonlfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        obj = {
            "instruction": row["question"],
            "input": "",
            "output": row["answer"]
        }
        jsonlfile.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ Converted {input_csv} to {output_jsonl}")
