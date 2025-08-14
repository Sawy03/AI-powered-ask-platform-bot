import pandas as pd
import os
import shutil
from vector2 import vector_store, embeddings
from langchain_core.documents import Document

class KnowledgeBaseManager:
    """Utility class for managing the knowledge base"""
    
    def __init__(self, csv_path="all_documents.csv", db_location="./chroma_langchain_db"):
        self.csv_path = csv_path
        self.db_location = db_location
        self.vector_store = vector_store
    
    def update_knowledge_base(self, new_csv_path=None):
        """Update the knowledge base with new documents"""
        csv_path = new_csv_path or self.csv_path
        
        print("Updating knowledge base...")
        
        # Remove existing database
        if os.path.exists(self.db_location):
            shutil.rmtree(self.db_location)
            print("Removed existing database")
        
        # Recreate database with new documents
        os.system("python vector.py")  # This will recreate the database
        print("Knowledge base updated successfully!")
    
    def search_documents(self, query, k=5):
        """Search for documents matching the query"""
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        results = retriever.invoke(query)
        return results
    
    def get_stats(self):
        """Get statistics about the knowledge base"""
        try:
            df = pd.read_csv(self.csv_path)
            
            stats = {
                "total_rows": len(df),
                "estimated_chunks": len(df) // 6,
                "unique_sources": df['filename'].nunique() if 'filename' in df.columns else "Unknown",
                "database_exists": os.path.exists(self.db_location)
            }
            
            return stats
        except Exception as e:
            return {"error": str(e)}
    
    def preview_chunks(self, num_chunks=3):
        """Preview the first few chunks"""
        df = pd.read_csv(self.csv_path)
        
        print(f"Preview of first {num_chunks} chunks:")
        print("=" * 60)
        
        for i in range(0, min(num_chunks * 6, len(df)), 6):
            chunk = df.iloc[i:i+6]
            chunk_text = "\n".join(chunk["text"].dropna().astype(str)).strip()
            source = chunk["filename"].iloc[0] if "filename" in df.columns else "Unknown"
            
            print(f"\nChunk {i//6 + 1} (Source: {source}):")
            print("-" * 40)
            print(chunk_text[:200] + ("..." if len(chunk_text) > 200 else ""))
            print()

def main():
    """Command line interface for knowledge base management"""
    import sys
    
    kb_manager = KnowledgeBaseManager()
    
    if len(sys.argv) < 2:
        print("Usage: python kb_utils.py [command]")
        print("Commands:")
        print("  stats     - Show knowledge base statistics")
        print("  preview   - Preview document chunks")
        print("  update    - Update knowledge base")
        print("  search    - Search documents (interactive)")
        return
    
    command = sys.argv[1].lower()
    
    if command == "stats":
        stats = kb_manager.get_stats()
        print("Knowledge Base Statistics:")
        print("=" * 30)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    elif command == "preview":
        kb_manager.preview_chunks()
    
    elif command == "update":
        kb_manager.update_knowledge_base()
    
    elif command == "search":
        query = input("Enter search query: ")
        results = kb_manager.search_documents(query)
        
        print(f"\nFound {len(results)} results:")
        print("=" * 50)
        
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            print(f"\n{i}. Source: {source}")
            print("-" * 30)
            print(doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""))
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()