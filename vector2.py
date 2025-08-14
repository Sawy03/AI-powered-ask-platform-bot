from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load your new Q&A CSV file
df = pd.read_csv("ai_generated_qa_dataset.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

def create_qa_chunks_from_csv(df):
    """
    Create chunks from Q&A CSV data
    Each Q&A pair becomes a document chunk
    """
    documents = []
    ids = []
    
    for index, row in df.iterrows():
        # For simple format
        if 'question' in df.columns and 'answer' in df.columns:
            question = str(row['question']).strip()
            answer = str(row['answer']).strip()
            source = row.get('source', 'unknown')
            category = row.get('category', 'general')
            
            # Combine question and answer for better context
            combined_text = f"Q: {question}\n\nA: {answer}"
            
            if len(combined_text.strip()) < 20:
                continue
                
            doc_id = f"qa_{index}"
            
            metadata = {
                "source": source,
                "category": category,
                "question": question,
                "answer": answer,
                "qa_pair_id": index
            }
            
        # For detailed Slack-like format  
        elif 'message_type' in df.columns:
            text_content = str(row['text']).strip()
            message_type = row.get('message_type', 'unknown')
            user = row.get('user', 'unknown')
            source = row.get('source_document', 'unknown')
            
            if len(text_content.strip()) < 10:
                continue
                
            doc_id = f"message_{index}"
            combined_text = text_content
            
            metadata = {
                "source": source,
                "message_type": message_type,
                "user": user,
                "timestamp": row.get('timestamp', ''),
                "thread_id": row.get('thread_id', ''),
                "message_id": index
            }
        
        else:
            continue
            
        document = Document(
            page_content=combined_text,
            metadata=metadata,
            id=doc_id
        )
        
        ids.append(doc_id)
        documents.append(document)
    
    return documents, ids

# Initialize vector store
vector_store = Chroma(
    collection_name="qa_documents",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if add_documents:
    print("Creating Q&A chunks...")
    documents, ids = create_qa_chunks_from_csv(df)
    print(f"Created {len(documents)} Q&A chunks")
    
    # Add documents to vector store
    vector_store.add_documents(documents=documents, ids=ids)
    print("Q&A documents added to vector store")

# Create retriever with similarity threshold
retrieval = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,  # Get more results since Q&A pairs are more focused
        "score_threshold": 0.5  # Lower threshold for Q&A matching
    }
)