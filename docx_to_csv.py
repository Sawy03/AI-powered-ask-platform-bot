import os
import pandas as pd
from docx import Document

# Set your folder path
folder_path = "word2"

# Extract text from each .docx file
data = []
for filename in os.listdir(folder_path):
    if filename.endswith(".docx"):
        doc_path = os.path.join(folder_path, filename)
        doc = Document(doc_path)
        text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
        data.append({"filename": filename, "text": text})

# Save as CSV or TXT
df = pd.DataFrame(data)
df.to_csv("all_documents2.csv", index=False)

# # Optional: Save all text into one clean .txt
# with open("all_documents.txt", "w", encoding="utf-8") as f:
#     for row in data:
#         f.write(f"=== {row['filename']} ===\n{row['text']}\n\n")

