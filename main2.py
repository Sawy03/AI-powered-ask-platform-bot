from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector2 import retrieval, vector_store
import json

model = OllamaLLM(model="llama3.2:1b")

# Improved template with better instructions
template = """
You are an expert AI assistant for a platform team's knowledge base. Your role is to answer questions based ONLY on the provided context from internal documents.

IMPORTANT INSTRUCTIONS:
1. Answer ONLY using information from the provided context
2. If the context contains a direct answer, provide it clearly and cite the source
3. If the context doesn't contain enough information for a complete answer, say "I don't have enough information in the knowledge base to fully answer this question, but here's what I found that might be related:"
4. Always mention the source document when possible
5. Be concise but complete
6. If no relevant information is found, say "I couldn't find relevant information in the knowledge base for this question"
7. Please if you don't have enough information, say so clearly
8. Please do not make up information or provide guesses
9. Please if you are not sure about the answer say that you are not sure

Context from knowledge base:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def format_context(retrieved_docs):
    """Format retrieved documents for better readability"""
    if not retrieved_docs:
        return "No relevant information found."
    
    formatted_context = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        chunk_id = doc.metadata.get('chunk_id', 'Unknown')
        
        formatted_context.append(f"Document {i} (Source: {source}, Chunk: {chunk_id}):\n{doc.page_content}\n")
    
    return "\n".join(formatted_context)

def check_relevance_score(query, retrieved_docs, min_score=0.3):
    """Check if the retrieved documents meet minimum relevance threshold"""
    if not retrieved_docs:
        return False, "No documents retrieved"
    
    # This is a simple check - you might want to implement more sophisticated relevance scoring
    return True, "Documents found"

def main():
    print("Platform Knowledge Base Bot")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        print("\n" + "-" * 60)
        user_input = input("Ask a question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not user_input:
            print("Please enter a question.")
            continue
            
        try:
            print("\nSearching knowledge base...")
            
            # Retrieve relevant documents
            retrieved_docs = retrieval.invoke(user_input)
            print(f"Found {len(retrieved_docs)} relevant documents.")
            print(retrieved_docs)
            
            # Format context for display
            context_display = format_context(retrieved_docs)
            
            # Check if we found relevant information
            has_relevant_info, relevance_msg = check_relevance_score(user_input, retrieved_docs)
            
            if not retrieved_docs:
                print("\n" + "=" * 60)
                print("RESPONSE:")
                print("I couldn't find any relevant information in the knowledge base for this question. Please check with the platform team directly or rephrase your question.")
                continue
            
            # Generate response
            context_for_llm = "\n\n".join([doc.page_content for doc in retrieved_docs])
            result = chain.invoke({"context": context_for_llm, "question": user_input})
            
            # Display results
            print("\n" + "=" * 60)
            print("RESPONSE:")
            print(result)
            
            print("\n" + "-" * 40)
            print("SOURCES CONSULTED:")
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                chunk_id = doc.metadata.get('chunk_id', 'Unknown')
                print(f"{i}. {source} (Chunk {chunk_id})")
            
            # Optionally show context (for debugging)
            show_context = input("\nShow retrieved context? (y/n): ").lower().strip()
            if show_context == 'y':
                print("\n" + "-" * 40)
                print("RETRIEVED CONTEXT:")
                print(context_display)
                
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            print("Please try rephrasing your question or contact the @ask-platform team.")

if __name__ == "__main__":
    main()