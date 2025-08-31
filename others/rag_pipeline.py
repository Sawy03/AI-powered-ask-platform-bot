# rag_pipeline.py

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector2 import retrieval  # Your existing retrieval logic


# Initialize model
model = OllamaLLM(model="llama3.2:1b")

# Create prompt template
template = """
You are a helpful AI assistant for the platform team's knowledge base. Answer questions based ONLY on the provided context.

Be concise and helpful. If you don't have enough information, say so clearly.

Context: {context}
Question: {question}

Answer:"""

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

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

prompt_with_context = ChatPromptTemplate.from_template(template_with_context)
prompt_no_context = ChatPromptTemplate.from_template(template_no_context)

chain_with_context = prompt_with_context | model
chain_no_context = prompt_no_context | model

def get_bot_response_with_context(message, thread_context=""):
    """Get response from RAG pipeline with optional thread context"""
    try:
        print(f"Processing message: {message}")
        if thread_context:
            print(f"Thread context length: {len(thread_context)} characters")
        
        # Get documents from your existing retrieval
        docs = retrieval.invoke(message)
        
        if not docs:
            if thread_context:
                # Even without knowledge base docs, we can respond using thread context
                return f"I don't have specific information in the knowledge base for this question, but based on our conversation, I can see we were discussing related topics. Please contact the platform team for more detailed information."
            else:
                return "Sorry, I couldn't find relevant information in the knowledge base for your question. Please contact the platform team directly."
        
        # Prepare context (limit to prevent too long responses)
        context_parts = []
        for doc in docs[:5]:  # Limit to 5 most relevant docs
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content[:400]  # Limit content length for Slack
            context_parts.append(f"Source: {source}\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Generate response using appropriate chain
        if thread_context.strip():
            # Use context-aware prompt
            result = chain_with_context.invoke({
                "thread_context": thread_context,
                "context": context, 
                "question": message
            })
        else:
            # Use regular prompt
            result = chain_no_context.invoke({
                "context": context, 
                "question": message
            })
        
        # Add source information
        sources = [doc.metadata.get('source', 'Unknown') for doc in docs[:3]]
        unique_sources = list(set(sources))
        
        if len(unique_sources) > 0:
            sources_text = f"\n\n📚 Sources: {', '.join(unique_sources[:2])}"
            result += sources_text
        
        return result
        
    except Exception as e:
        print(f"Error in get_bot_response_with_context: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"

def get_bot_response(message):
    """Get response from RAG pipeline"""
    try:
        print(f"Processing message: {message}")
        
        # Get documents from your existing retrieval
        docs = retrieval.invoke(message)
        
        if not docs:
            return "Sorry, I couldn't find relevant information in the knowledge base for your question. Please contact the platform team directly."
        
        # Prepare context (limit to prevent too long responses)
        context_parts = []
        for doc in docs[:5]:  # Limit to 5 most relevant docs
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content[:400]  # Limit content length for Slack
            context_parts.append(f"Source: {source}\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Generate response using your model
        result = chain.invoke({"context": context, "question": message})
        
        # Add source information
        sources = [doc.metadata.get('source', 'Unknown') for doc in docs[:3]]
        unique_sources = list(set(sources))
        
        if len(unique_sources) > 0:
            sources_text = f"\n\n📚 Sources: {', '.join(unique_sources[:2])}"
            result += sources_text
        
        return result
        
    except Exception as e:
        print(f"Error in get_bot_response: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"