import faiss
import numpy as np


class FaissVectorStore:
    """
    A vector store using FAISS for efficient similarity search.
    """

    def __init__(self, embedding_dim):
        """
        Initialize the FAISS index with the embedding dimension.

        Args:
            embedding_dim (int): The size of the embedding vectors.
        """
        self.index = faiss.IndexFlatIP(
            embedding_dim
        )  # Inner product for cosine similarity
        self.passages = []

    def add(self, embeddings, passages):
        """
        Add embeddings and their corresponding passages to the store.

        Args:
            embeddings (np.ndarray): Array of shape (n, dim) with embeddings.
            passages (list): List of passage strings.
        """
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        self.index.add(normalized_embeddings)
        self.passages.extend(passages)

    def search(self, query_embedding, k):
        """
        Search for the top-k most similar passages.

        Args:
            query_embedding (np.ndarray): The query embedding.
            k (int): Number of passages to retrieve.

        Returns:
            list: List of retrieved passage strings.
        """
        # Normalize query embedding
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        distances, indices = self.index.search(np.array([normalized_query]), k)
        return [self.passages[i] for i in indices[0]]
