"""
One-time cleanup script for confident Q&A database and vector store
Run this script once to fix the current issues
"""

import os
import sys
from dotenv import load_dotenv

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_qa_tracker import SmartQATracker

def main():
    print("🧹 Starting confident Q&A cleanup process...")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize Smart Q&A Tracker
    tracker = SmartQATracker(
        base_url=os.getenv("CONFLUENCE_BASE_URL"),
        username=os.getenv("CONFLUENCE_USERNAME"),
        api_token=os.getenv("CONFLUENCE_API_TOKEN"),
        space_keys=os.getenv("CONFLUENCE_SPACE_KEYS", "").split(",") if os.getenv("CONFLUENCE_SPACE_KEYS") else None
    )
    
    print("\n1. Cleaning database of invalid entries...")
    tracker.clean_confident_database()
    
    print("\n2. Recreating confident vector store...")
    tracker.recreate_confident_vector_store()
    
    print("\n3. Testing confident retriever...")
    try:
        retriever = tracker.get_confident_retriever()
        test_docs = retriever.invoke("test question")
        print(f"✅ Test successful - found {len(test_docs)} documents")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\n4. Showing database summary...")
    # Show confident Q&A pairs
    confident_pairs = tracker.get_confident_qa_pairs()
    print(f"📊 Total confident Q&A pairs: {len(confident_pairs)}")
    
    for pair in confident_pairs[:5]:  # Show first 5
        q = pair.get('question', 'No question')[:50]
        a = pair.get('answer', 'No answer')[:50]
        print(f"  - Q: {q}...")
        print(f"    A: {a}...")
    
    if len(confident_pairs) > 5:
        print(f"    ... and {len(confident_pairs) - 5} more pairs")
    
    print("\n✅ Cleanup completed successfully!")
    print("\nYour system should now work without the validation errors.")

if __name__ == "__main__":
    main()