from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from smart_qa_tracker import SmartQATracker
import os
from dotenv import load_dotenv
from prompts import template_no_context, template_with_context
load_dotenv()


# Initialize Smart Q&A Tracker
smart_tracker = SmartQATracker(
    base_url=os.getenv("CONFLUENCE_BASE_URL"),
    username=os.getenv("CONFLUENCE_USERNAME"),
    api_token=os.getenv("CONFLUENCE_API_TOKEN"),
    space_keys=os.getenv("CONFLUENCE_SPACE_KEYS", "").split(",") if os.getenv("CONFLUENCE_SPACE_KEYS") else None
)


# Get retriever from Smart Tracker for confluence Q&A
retrieval = smart_tracker.get_retriever(
    k=5,
    score_threshold=0.6
)


# NEW: Get retriever for confident Q&A
confident_retrieval = smart_tracker.get_confident_retriever(
    k=3,
    score_threshold=0.6
)


# Initialize model
model = OllamaLLM(model="llama3.2:1b")


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
        confident_score = doc.metadata.get('confidence_score', 1)
        page_title = doc.metadata.get('page_title', 'Unknown Document')
        space_name = doc.metadata.get('space_name', 'Unknown Space')
        print(question)
        print(answer)
        
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
            Confidence Score: {confident_score}
            """
        context_parts.append(context_part.strip())
    
    return "\n\n" + "="*50 + "\n\n".join(context_parts)


def get_bot_response_with_context(message, thread_context=""):
    """Get response from Q&A RAG pipeline with priority on confident database"""
    try:
        print(f"🔍 Processing Q&A query: {message}")
        
        # Step 1: First try confident Q&A retriever
        print("🎯 Searching confident Q&A database...")
        confident_docs = []
        confident_retrieval_failed = False
        
        try:
            confident_retrieval_instance = smart_tracker.get_confident_retriever()
            if confident_retrieval_instance:
                confident_docs = confident_retrieval_instance.invoke(message)
        except Exception as e:
            print(f"❌ Error with confident retrieval: {e}")
            confident_retrieval_failed = True
            
            # Try to fix the issue
            try:
                print("🔧 Attempting to fix confident database...")
                smart_tracker.clean_confident_database()
                smart_tracker.recreate_confident_vector_store()
                
                # Retry after fixing
                confident_retrieval_instance = smart_tracker.get_confident_retriever()
                if confident_retrieval_instance:
                    confident_docs = confident_retrieval_instance.invoke(message)
                    print("✅ Fixed confident database and retry successful")
            except Exception as fix_error:
                print(f"❌ Failed to fix confident database: {fix_error}")
                confident_docs = []
        
        if confident_docs and len(confident_docs) > 0:
            print(f"✅ Found {len(confident_docs)} confident Q&A pairs")
            confident_docs.sort(key=lambda doc: doc.metadata.get('confidence_score', 0), reverse=True)
            docs = confident_docs
            context = format_qa_context(docs)
            print("🚀 Using confident database results")
            print("This is what ollama gets: " + context)
        else:
            # Step 2: If no confident results, try confluence database
            print("🔍 No confident results found, searching confluence database...")
            try:
                docs = retrieval.invoke(message)
                
                if not docs:
                    if thread_context:
                        return "I don't have enough information, but based on our conversation, I can see we were discussing related topics. Please contact <@U099C4LNDEC> for more detailed information."
                    else:
                        return "Sorry, I don't have enough information. Please contact <@U099C4LNDEC> directly."
                
                context = format_qa_context(docs[:5])
                print(f"📚 Found {len(docs)} relevant confluence Q&A pairs")
            except Exception as confluence_error:
                print(f"❌ Error with confluence retrieval: {confluence_error}")
                return "Sorry, I'm having trouble accessing the knowledge base right now. Please contact <@U099C4LNDEC> for assistance."
        
        # Generate response using appropriate chain
        try:
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
        except Exception as llm_error:
            print(f"❌ Error generating response: {llm_error}")
            return "Sorry, I encountered an error generating the response. Please contact <@U099C4LNDEC> for assistance."
        
        sources = []
        seen_pages = set()
        
        for doc in docs[:3]:
            try:
                if doc.metadata.get('type') == 'confident_qa':
                    source_info = ""
                else:
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
            except Exception as source_error:
                print(f"⚠️ Error processing source info: {source_error}")
                continue
        print("result : " + result)
        if sources and doc.metadata.get('type') != 'confident_qa':
            sources_text = f"\n\n📚 **Source Documents:**\n" + "\n".join([f"• {source}" for source in sources])
            result += sources_text

        text = "Help us get better, please use the following emojis for answer: \n\n👍 upvotes \n\n👎 downvotes \n\nDisclaimer: If you have more questions related to the same topic please mention me in the thread using <@U099VBD9BR7>"
        result += "\n\n" + text
        return result
        
    except Exception as e:
        print(f"❌ Error in Q&A response generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Sorry, I encountered an error processing your question. Please contact <@U099C4LNDEC> for assistance."


def initialize_confident_qa_vector_store():
    """Initialize confident Q&A vector store with existing data and cleanup"""
    try:
        print("🔄 Initializing confident Q&A vector store...")
        
        # Clean database first
        smart_tracker.clean_confident_database()
        
        # Recreate vector store
        smart_tracker.recreate_confident_vector_store()
        
        print("✅ Confident Q&A vector store initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing confident Q&A vector store: {e}")
        import traceback
        traceback.print_exc()


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
    
    # Initialize confident Q&A vector store
    initialize_confident_qa_vector_store()
    
    # Initialize Q&A data with smart tracking (only process changed pages)
    initialize_confluence_qa_data(force_regenerate=False)
    
    # Show tracking summary
    show_qa_tracking_summary()