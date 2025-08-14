from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

df = pd.read_csv("all_documents.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = ".\chroma_langchain_db"
add_documents = not os.path.exists(db_location)


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

if add_documents:
    documents = []
    ids = []

    # Loop through every 6 rows
    for i in range(0, len(df), 6):
        chunk = df.iloc[i:i+6]  # Get a group of 6 rows
        patch_text = "\n".join(chunk["text"].dropna().astype(str)).strip()  # Combine text

        if patch_text:  # Skip empty patches
            doc_id = str(i // 6)  # Unique ID per patch
            source = chunk["filename"].iloc[0] if "filename" in df.columns else "unknown"

            document = Document(
                page_content=patch_text,
                metadata={"source": source},
                id=doc_id
            )
            ids.append(doc_id)
            documents.append(document)


vector_store = Chroma(
    collection_name="documents",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)


retrieval = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)