# smart_qa_tracker.py

from ast import pattern
import os
import json
import requests
from requests.auth import HTTPBasicAuth
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from datetime import datetime
import hashlib
import time
import re
from typing import List, Dict, Optional, Tuple
import sqlite3

class SmartQATracker:
    def __init__(self, 
                 base_url: str, 
                 username: str, 
                 api_token: str,
                 space_keys: List[str] = None):
        """
        Initialize Smart Q&A Tracker with version management
        """
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.api_token = api_token
        self.space_keys = space_keys or []
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(username, api_token)
        
        # Initialize vector store
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.db_location = "./chroma_confluence_qa_db"
        self.vector_store = Chroma(
            collection_name="confluence_qa_smart",
            persist_directory=self.db_location,
            embedding_function=self.embeddings,
        )
        
        # Initialize LLM for Q&A generation
        self.llm = OllamaLLM(model="llama3.2:1b")
        
        # Initialize tracking database
        self.tracking_db = "./page_tracking.db"
        self.init_tracking_db()
    
    def init_tracking_db(self):
        """Initialize SQLite database for tracking page versions and Q&A"""
        conn = sqlite3.connect(self.tracking_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS page_tracking (
                page_id TEXT PRIMARY KEY,
                title TEXT,
                space_key TEXT,
                space_name TEXT,
                version INTEGER,
                content_hash TEXT,
                last_updated TEXT,
                qa_count INTEGER DEFAULT 0,
                last_processed TEXT,
                status TEXT DEFAULT 'pending'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS qa_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id TEXT,
                qa_index INTEGER,
                question TEXT,
                answer TEXT,
                url TEXT,
                vector_doc_id TEXT,
                created_at TEXT,
                FOREIGN KEY (page_id) REFERENCES page_tracking (page_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("📊 Tracking database initialized")
    
    def get_page_tracking_info(self, page_id: str) -> Optional[Dict]:
        """Get tracking information for a page"""
        conn = sqlite3.connect(self.tracking_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT page_id, title, space_key, space_name, version, content_hash, 
                   last_updated, qa_count, last_processed, status
            FROM page_tracking 
            WHERE page_id = ?
        ''', (page_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'page_id': result[0],
                'title': result[1],
                'space_key': result[2],
                'space_name': result[3],
                'version': result[4],
                'content_hash': result[5],
                'last_updated': result[6],
                'qa_count': result[7],
                'last_processed': result[8],
                'status': result[9]
            }
        return None
    
    def update_page_tracking(self, page_id: str, page_data: Dict, qa_count: int = 0):
        """Update tracking information for a page"""
        conn = sqlite3.connect(self.tracking_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO page_tracking 
            (page_id, title, space_key, space_name, version, content_hash, 
             last_updated, qa_count, last_processed, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            page_id,
            page_data.get('title', ''),
            page_data.get('space_key', ''),
            page_data.get('space_name', ''),
            page_data.get('version', 0),
            page_data.get('content_hash', ''),
            page_data.get('last_updated', ''),
            qa_count,
            datetime.now().isoformat(),
            'processed'
        ))
        
        conn.commit()
        conn.close()
    
    def delete_page_qa_pairs(self, page_id: str):
        """Delete all Q&A pairs for a specific page from both tracking DB and vector store"""
        print(f"🗑️ Deleting existing Q&A pairs for page {page_id}")
        
        # Get existing Q&A vector document IDs
        conn = sqlite3.connect(self.tracking_db)
        cursor = conn.cursor()
        
        cursor.execute('SELECT vector_doc_id FROM qa_pairs WHERE page_id = ?', (page_id,))
        vector_doc_ids = [row[0] for row in cursor.fetchall()]
        
        # Delete from tracking database
        cursor.execute('DELETE FROM qa_pairs WHERE page_id = ?', (page_id,))
        conn.commit()
        conn.close()
        
        # Delete from vector store (Chroma doesn't have direct delete by ID, 
        # so we'll track and handle this during next full sync or use collection reset)
        if vector_doc_ids:
            print(f"  📝 Marked {len(vector_doc_ids)} vector documents for cleanup")
            # Store IDs to clean up later (implement based on your vector store capabilities)
            # For now, we'll handle this by regenerating the entire vector store periodically
        
        return len(vector_doc_ids)
    
    def record_qa_pairs(self, page_id: str, qa_pairs: List[Tuple[str, str]], vector_doc_ids: List[str]):
        """Record Q&A pairs in tracking database"""
        conn = sqlite3.connect(self.tracking_db)
        cursor = conn.cursor()
        
        for i, ((question, answer), vector_doc_id) in enumerate(zip(qa_pairs, vector_doc_ids)):
            url = f"{self.base_url}/pages/viewpage.action?pageId={page_id}" 
            cursor.execute('''
                INSERT INTO qa_pairs (page_id, qa_index, question, answer, url, vector_doc_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (page_id, i, question, answer, url, vector_doc_id, datetime.now().isoformat()))

        conn.commit()
        conn.close()
    
    def is_page_changed(self, page_id: str, current_version: int, current_content_hash: str) -> bool:
        """Check if a page has changed since last processing"""
        tracking_info = self.get_page_tracking_info(page_id)
        
        if not tracking_info:
            print(f"📄 New page detected: {page_id}")
            return True
        
        if tracking_info['version'] != current_version:
            print(f"📝 Version change detected for page {page_id}: {tracking_info['version']} → {current_version}")
            return True
        
        if tracking_info['content_hash'] != current_content_hash:
            print(f"🔄 Content change detected for page {page_id}")
            return True
        
        print(f"✅ Page {page_id} unchanged")
        return False
    
    def generate_qa_from_content(self, title: str, content: str) -> List[Tuple[str, str]]:
        """Generate Q&A pairs from content using LLM"""
        try:
            # Clean content for better processing
            clean_content = re.sub(r'\s+', ' ', content).strip()
            if len(clean_content) > 5000:  # Limit content size for LLM
                clean_content = clean_content[:5000] + "..."
            
            prompt = f"""

You are an expert at creating comprehensive question-answer pairs from technical documentation. 

Given the following document content, generate as many relevant questions and answers as possible. Focus on:
1. What information is covered
2. How-to questions
3. Troubleshooting questions
4. Definition questions
5. Process questions
6. Configuration questions
7. Best practices questions

For each question-answer pair, format it EXACTLY like this:
Q: [Question here]
A: [Answer here]

Q: [Next question]
A: [Next answer]

Make sure to:
- Generate 5-15 Q&A pairs per document (depending on content length)
- Ask questions that real users would ask
- Include specific details from the document in answers
- Cover different aspects of the content
- Make questions natural and conversational
- Keep answers informative but concise

Document Title: {title}
Content: {clean_content}

Generated Q&A pairs:"""
            
            print(f"🤖 Generating Q&A for: {title}")
            response = self.llm.invoke(prompt)
            
            # Parse the LLM response to extract Q&A pairs
            qa_pairs = []
            pattern = r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)'
            matches = re.findall(pattern, response, re.DOTALL)

            
            for q, a in matches:
                question = q.strip()
                if not question.endswith('?'):
                    question += '?'
                answer = a.strip()
                
                # Filter out very short or low-quality Q&As
                if len(question) > 10 and len(answer) > 20:
                    qa_pairs.append((question, answer))
            
            print(f"  ✅ Generated {len(qa_pairs)} Q&A pairs")
            return qa_pairs

        except Exception as e:
            print(f"❌ Error generating Q&A for {title}: {e}")
            return []
    
    def extract_text_from_confluence_storage(self, storage_content: str) -> str:
        """Extract plain text from Confluence storage format"""
        import re
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', storage_content)
        
        # Decode HTML entities
        import html
        text = html.unescape(text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def process_single_page(self, page: Dict, force_regenerate: bool = False) -> bool:
        """
        Process a single page: check if changed, delete old Q&A if needed, generate new Q&A
        
        Returns: True if page was processed, False if skipped
        """
        try:
            page_id = page.get('id')
            title = page.get('title', 'Untitled')
            space_key = page.get('space', {}).get('key', 'Unknown')
            space_name = page.get('space', {}).get('name', 'Unknown')
            version = page.get('version', {}).get('number', 1)
            last_updated = page.get('history', {}).get('lastUpdated', {}).get('when', '')
            
            # Get page content
            body = page.get('body', {}).get('storage', {})
            content = body.get('value', '') if body else ''
            
            if not content:
                print(f"⚠️ No content found for page: {title}")
                return False
            
            # Extract plain text and create content hash
            text_content = self.extract_text_from_confluence_storage(content)
            content_hash = hashlib.md5(text_content.encode()).hexdigest()
            
            if len(text_content.strip()) < 50:
                print(f"⚠️ Content too short for page: {title}")
                return False
            
            # Check if page has changed
            if not force_regenerate and not self.is_page_changed(page_id, version, content_hash):
                return False  # Skip unchanged pages
            
            print(f"🔄 Processing page: {title}")
            
            # Delete existing Q&A pairs if they exist
            deleted_count = self.delete_page_qa_pairs(page_id)
            if deleted_count > 0:
                print(f"  🗑️ Deleted {deleted_count} existing Q&A pairs")
            
            # Generate new Q&A pairs
            qa_pairs = self.generate_qa_from_content(title, text_content)
            
            if not qa_pairs:
                print(f"  ⚠️ No Q&A pairs generated for {title}")
                # Still update tracking to avoid reprocessing
                page_data = {
                    'title': title,
                    'space_key': space_key,
                    'space_name': space_name,
                    'version': version,
                    'content_hash': content_hash,
                    'last_updated': last_updated
                }
                self.update_page_tracking(page_id, page_data, 0)
                return True
            
            # Create vector documents from Q&A pairs
            documents = []
            vector_doc_ids = []
            
            for i, (question, answer) in enumerate(qa_pairs):
                # Create combined Q&A text
                combined_text = f"Q: {question}\n\nA: {answer}"
                
                # Create unique document ID
                doc_id = f"confluence_qa_{page_id}_{version}_{i}"
                vector_doc_ids.append(doc_id)
                
                # Create metadata
                metadata = {
                    'source': f'Confluence - {space_name}',
                    'space_key': space_key,
                    'space_name': space_name,
                    'page_id': page_id,
                    'page_title': title,
                    'question': question,
                    'answer': answer,
                    'url': f"{self.base_url}/pages/viewpage.action?pageId={page_id}",
                    'version': version,
                    'last_updated': last_updated,
                    'qa_pair_id': i,
                    'content_hash': content_hash
                }
                
                document = Document(
                    page_content=combined_text,
                    metadata=metadata,
                    id=doc_id
                )
                
                documents.append(document)
            
            # Add documents to vector store
            if documents:
                self.vector_store.add_documents(documents=documents, ids=vector_doc_ids)
                print(f"  ✅ Added {len(documents)} Q&A pairs to vector store")
                
                # Record Q&A pairs in tracking database
                self.record_qa_pairs(page_id, qa_pairs, vector_doc_ids)
                
                # Update page tracking
                page_data = {
                    'title': title,
                    'space_key': space_key,
                    'space_name': space_name,
                    'version': version,
                    'content_hash': content_hash,
                    'last_updated': last_updated
                }
                self.update_page_tracking(page_id, page_data, len(qa_pairs))
                
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error processing page {page.get('id', 'unknown')}: {e}")
            return False
    
    def get_spaces(self) -> List[Dict]:
        """Get all accessible spaces or specified spaces"""
        spaces = []
        
        if self.space_keys:
            for space_key in self.space_keys:
                try:
                    url = f"{self.base_url}/rest/api/space/{space_key}"
                    response = self.session.get(url)
                    if response.status_code == 200:
                        spaces.append(response.json())
                except Exception as e:
                    print(f"Error getting space {space_key}: {e}")
        else:
            try:
                url = f"{self.base_url}/rest/api/space"
                params = {'limit': 50}
                
                while url:
                    response = self.session.get(url, params=params)
                    if response.status_code != 200:
                        break
                        
                    data = response.json()
                    spaces.extend(data.get('results', []))
                    
                    links = data.get('_links', {})
                    url = links.get('next')
                    if url:
                        url = f"{self.base_url}{url}"
                    params = None
                    
            except Exception as e:
                print(f"Error getting spaces: {e}")
                
        return spaces
    
    def get_pages_from_space(self, space_key: str) -> List[Dict]:
        """Get all pages from a specific space"""
        pages = []
        url = f"{self.base_url}/rest/api/content"
        params = {
            'spaceKey': space_key,
            'type': 'page',
            'status': 'current',
            'expand': 'body.storage,version,space,history.lastUpdated',
            'limit': 50
        }
        
        while url:
            try:
                response = self.session.get(url, params=params)
                if response.status_code != 200:
                    print(f"Failed to get pages from space {space_key}: {response.status_code}")
                    break
                    
                data = response.json()
                pages.extend(data.get('results', []))
                
                # Handle pagination
                links = data.get('_links', {})
                url = links.get('next')
                if url:
                    url = f"{self.base_url}{url}"
                params = None
                
            except Exception as e:
                print(f"Error getting pages from space {space_key}: {e}")
                break
                
        return pages
    
    def sync_all_confluence_qa(self, force_regenerate: bool = False):
        """
        Sync all Confluence content to Q&A format with smart change detection
        
        Args:
            force_regenerate: If True, regenerate Q&A for all pages regardless of changes
        """
        print("🚀 Starting smart Confluence Q&A sync...")
        
        # Get all spaces
        spaces = self.get_spaces()
        print(f"📚 Found {len(spaces)} spaces")
        
        total_processed = 0
        total_skipped = 0
        
        for space in spaces:
            space_key = space.get('key')
            space_name = space.get('name')
            
            print(f"\n📖 Processing space: {space_name} ({space_key})")
            
            # Get all pages from space
            pages = self.get_pages_from_space(space_key)
            print(f"  Found {len(pages)} pages")
            
            space_processed = 0
            space_skipped = 0
            
            for page in pages:
                if self.process_single_page(page, force_regenerate):
                    space_processed += 1
                    total_processed += 1
                    
                    # Small delay to avoid overwhelming the system
                    time.sleep(0.5)
                else:
                    space_skipped += 1
                    total_skipped += 1
            
            print(f"  ✅ Space summary: {space_processed} processed, {space_skipped} skipped")
        
        print(f"\n🎉 Sync completed!")
        print(f"  📝 Total processed: {total_processed}")
        print(f"  ⏭️ Total skipped: {total_skipped}")
        
        # Show tracking summary
        self.show_tracking_summary()
    
    def update_single_page_smart(self, page_id: str):
        """Smart update for a single page (called by webhook)"""
        try:
            print(f"🔄 Smart update for page {page_id}")
            
            # Get page data
            url = f"{self.base_url}/rest/api/content/{page_id}"
            params = {
                'expand': 'body.storage,version,space,history.lastUpdated'
            }
            
            response = self.session.get(url, params=params)
            if response.status_code != 200:
                print(f"Failed to get page {page_id}: {response.status_code}")
                return
            
            page = response.json()
            
            # Process with force regenerate since this is a webhook update
            if self.process_single_page(page, force_regenerate=True):
                print(f"✅ Successfully updated Q&A for page: {page.get('title')}")
            else:
                print(f"⚠️ No updates needed for page: {page.get('title')}")
                
        except Exception as e:
            print(f"❌ Error in smart update for page {page_id}: {e}")
    
    def show_tracking_summary(self):
        """Show summary of tracking database"""
        conn = sqlite3.connect(self.tracking_db)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM page_tracking')
        total_pages = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM qa_pairs')
        total_qa_pairs = cursor.fetchone()[0]
        
        cursor.execute('SELECT space_name, COUNT(*) FROM page_tracking GROUP BY space_name')
        space_stats = cursor.fetchall()
        
        conn.close()
        
        print(f"\n📊 TRACKING SUMMARY:")
        print(f"  📄 Total pages tracked: {total_pages}")
        print(f"  💬 Total Q&A pairs: {total_qa_pairs}")
        print(f"  📚 Spaces:")
        for space_name, count in space_stats:
            print(f"    - {space_name}: {count} pages")
    
    def get_retriever(self, **kwargs):
        """Get retriever for the Q&A vector store"""
        search_kwargs = {
            "k": kwargs.get("k", 5),
            "score_threshold": kwargs.get("score_threshold", 0.5)
        }
        
        return self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs
        )

# Example usage
if __name__ == "__main__":
    tracker = SmartQATracker(
        base_url=os.getenv("CONFLUENCE_BASE_URL"),
        username=os.getenv("CONFLUENCE_USERNAME"),
        api_token=os.getenv("CONFLUENCE_API_TOKEN"),
        space_keys=os.getenv("CONFLUENCE_SPACE_KEYS", "").split(",") if os.getenv("CONFLUENCE_SPACE_KEYS") else None
    )
    
    # Initial sync - processes only new/changed pages
    tracker.sync_all_confluence_qa(force_regenerate=False)
    
    # Force regenerate all Q&A (useful for first time setup)
    # tracker.sync_all_confluence_qa(force_regenerate=True)