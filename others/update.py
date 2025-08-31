import shutil
import os

db_path = "./chroma_langchain_db"

if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print("🗑 ChromaDB deleted successfully!")
else:
    print("⚠️ No ChromaDB found at that path.")
