# qa_rag_pipeline.py

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from smart_qa_tracker import SmartQATracker
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Smart Q&A Tracker
smart_tracker = SmartQATracker(
    base_url=os.getenv("CONFLUENCE_BASE_URL"),
    username=os.getenv("CONFLUENCE_USERNAME"),
    api_token=os.getenv("CONFLUENCE_API_TOKEN"),
    space_keys=os.getenv("CONFLUENCE_SPACE_KEYS", "").split(",") if os.getenv("CONFLUENCE_SPACE_KEYS") else None
)

# Get retriever from Smart Tracker
retrieval = smart_tracker.get_retriever(
    k=5,
    score_threshold=0.5
)

# Initialize model
model = OllamaLLM(model="llama3.2:1b")

# Updated prompts for Q&A format
template_with_context = """
You are a helpful AI assistant for the platform team's knowledge base. Answer questions based ONLY on the provided context.

IMPORTANT: This conversation is part of a thread. Below is the conversation history that you MUST consider when answering:

=== CONVERSATION HISTORY ===
{thread_context}
=== END CONVERSATION HISTORY ===

Use this conversation history to:
1. Understand the context and continuation of the discussion
2. Refer back to previous questions and answers when relevant
3. Provide follow-up answers that build on the conversation
4. Clarify or expand on previous responses if needed

Be concise and helpful. If you don't have enough information, say so clearly.

Knowledge Base Context: {context}
Current Question: {question}
put in your mind that I can not see the knowledge base i can only see this question {question} and this conversation history so i need a detailed sumarized answer i do not want you to list the context


Answer:"""

template_no_context = """
You are a helpful AI assistant for the platform team's knowledge base. Answer questions based ONLY on the provided context.

Be concise and helpful. If you don't have enough information, say so clearly.

Context: {context}
Question: {question}
put in your mind that I can not see the knowledge base i can only see this question {question} so i need a detailed sumarized answer i do not want you to list the context


Answer:"""

prompt_with_context = ChatPromptTemplate.from_template(template_with_context)
prompt_no_context = ChatPromptTemplate.from_template(template_no_context)

chain_with_context = prompt_with_context | model
chain_no_context = prompt_no_context | model

def format_qa_context(docs) -> str:
    """Format Q&A documents for the prompt context"""
    context_parts = []
    
    for i, doc in enumerate(docs, 1):
        # Extract Q&A from metadata if available
        question = doc.metadata.get('question', '')
        answer = doc.metadata.get('answer', '')
        page_title = doc.metadata.get('page_title', 'Unknown Document')
        space_name = doc.metadata.get('space_name', 'Unknown Space')
        
        # If no separate Q&A in metadata, use page content
        if not question or not answer:
            content = doc.page_content
            if content.startswith("Q:") and "\n\nA:" in content:
                parts = content.split("\n\nA:", 1)
                question = parts[0].replace("Q:", "").strip()
                answer = parts[1].strip() if len(parts) > 1 else ""
            else:
                question = f"Information from {page_title}"
                answer = content
        
        context_part = f"""
Document: {page_title} ({space_name})
Question: {question}
Answer: {answer}
"""
        context_parts.append(context_part.strip())
    
    return "\n\n" + "="*50 + "\n\n".join(context_parts)

def get_bot_response_with_context(message, thread_context=""):
    """Get response from Q&A RAG pipeline with optional thread context"""
    try:
        print(f"🔍 Processing Q&A query: {message}")
        if thread_context:
            print(f"🧵 Thread context length: {len(thread_context)} characters")
        
        # Get Q&A documents from retrieval
        docs = retrieval.invoke(message)
        
        if not docs:
            if thread_context:
                return "I don't have enough information, but based on our conversation, I can see we were discussing related topics. Please contact <@U099C4LNDEC> for more detailed information."
            else:
                return "Sorry, I don't have enough information. Please contact <@U099C4LNDEC> directly."

        # Format Q&A context
        context = format_qa_context(docs[:5])
        
        print(f"📚 Found {len(docs)} relevant Q&A pairs")
        
        # Generate response using appropriate chain
        if thread_context.strip():
            result = chain_with_context.invoke({
                "thread_context": thread_context,
                "context": context,
                "question": message
            })
        else:
            result = chain_no_context.invoke({
                "context": context,
                "question": message
            })
        
        # Add source information
        sources = []
        seen_pages = set()
        
        for doc in docs[:3]:
            page_title = doc.metadata.get('page_title', 'Unknown')
            space_name = doc.metadata.get('space_name', 'Unknown')
            url = doc.metadata.get('url', '')
            
            page_key = f"{page_title}-{space_name}"
            if page_key not in seen_pages:
                seen_pages.add(page_key)
                if url:
                    source_info = f"[{page_title}]({url}) - {space_name}"
                else:
                    source_info = f"{page_title} - {space_name}"
                sources.append(source_info)
        
        if sources:
            sources_text = f"\n\n📚 **Source Documents:**\n" + "\n".join([f"• {source}" for source in sources])
            result += sources_text
        
        return result
        
    except Exception as e:
        print(f"❌ Error in Q&A response generation: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"

def get_bot_response(message):
    """Get response from Q&A RAG pipeline (backward compatibility)"""
    return get_bot_response_with_context(message, "")

def initialize_confluence_qa_data(force_regenerate: bool = False):
    """
    Initialize Confluence Q&A data sync with smart tracking
    
    Args:
        force_regenerate: If True, regenerate Q&A for all pages regardless of changes
                         If False, only process new/changed pages (recommended)
    """
    try:
        print(f"🔄 Initializing Confluence Q&A data (force_regenerate={force_regenerate})")
        smart_tracker.sync_all_confluence_qa(force_regenerate=force_regenerate)
        print("✅ Confluence Q&A data initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing Confluence Q&A data: {e}")

def update_single_page_qa(page_id: str):
    """Update Q&A for a single page using smart tracking"""
    try:
        print(f"🔄 Updating Q&A for page {page_id}")
        smart_tracker.update_single_page_smart(page_id)
    except Exception as e:
        print(f"Error updating Q&A for page {page_id}: {e}")

def show_qa_tracking_summary():
    """Show summary of Q&A tracking"""
    smart_tracker.show_tracking_summary()

if __name__ == "__main__":
    # Test the Smart Q&A system
    print("🧪 Testing Smart Confluence Q&A integration...")
    
    # Initialize Q&A data with smart tracking (only process changed pages)
    initialize_confluence_qa_data(force_regenerate=False)
    
    # Show tracking summary
    show_qa_tracking_summary()
    
    # Test query
    test_response = get_bot_response("How do I deploy the application?")
    print(f"Test response: {test_response}")