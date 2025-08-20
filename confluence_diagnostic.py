#!/usr/bin/env python3
"""
Test specific Incorta space access and find more pages
"""
import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import json

load_dotenv()

def test_incorta_space_access():
    """Test access to the Incorta space specifically"""
    
    base_url = "https://incorta.atlassian.net/wiki"
    username = os.getenv("CONFLUENCE_USERNAME")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")
    
    print("🔍 TESTING INCORTA SPACE ACCESS")
    print("=" * 50)
    
    session = requests.Session()
    session.auth = HTTPBasicAuth(username, api_token.strip('"\''))
    
    # Test 1: Try to access Incorta space directly
    print("📁 Test 1: Accessing Incorta space...")
    try:
        space_url = f"{base_url}/rest/api/space/Incorta"
        response = session.get(space_url, timeout=10)
        
        if response.status_code == 200:
            space_data = response.json()
            print("✅ Direct access to Incorta space successful!")
            print(f"   Name: {space_data.get('name')}")
            print(f"   Key: {space_data.get('key')}")
            space_key = space_data.get('key')
        elif response.status_code == 404:
            print("❌ Incorta space not found with key 'Incorta'")
            # Try different possible space keys
            possible_keys = ['INC', 'Incorta', 'IN', 'TEAM']
            space_key = None
            
            for key in possible_keys:
                try:
                    test_url = f"{base_url}/rest/api/space/{key}"
                    test_response = session.get(test_url, timeout=5)
                    if test_response.status_code == 200:
                        space_data = test_response.json()
                        print(f"✅ Found space with key '{key}':")
                        print(f"   Name: {space_data.get('name')}")
                        space_key = key
                        break
                except:
                    continue
            
            if not space_key:
                print("❌ Could not find the Incorta space key")
                # Use the page to determine space key
                print("🔍 Getting space key from the known page...")
                page_url = f"{base_url}/rest/api/content/1763213371"
                page_response = session.get(page_url, params={'expand': 'space'}, timeout=10)
                if page_response.status_code == 200:
                    page_data = page_response.json()
                    space_info = page_data.get('space', {})
                    space_key = space_info.get('key')
                    space_name = space_info.get('name')
                    print(f"✅ Found space key from page: '{space_key}' ({space_name})")
                else:
                    print("❌ Could not determine space key")
                    return False
        else:
            print(f"❌ Error accessing space: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    if not space_key:
        return False
    
    # Test 2: Get pages from the Incorta space
    print(f"\n📄 Test 2: Getting pages from space '{space_key}'...")
    try:
        content_url = f"{base_url}/rest/api/content"
        params = {
            'spaceKey': space_key,
            'type': 'page',
            'status': 'current',
            'expand': 'space,version,ancestors'
        }
        
        response = session.get(content_url, params=params, timeout=15)
        
        if response.status_code == 200:
            content_data = response.json()
            pages = content_data.get('results', [])
            total = content_data.get('size', 0)
            
            print(f"✅ Found {len(pages)} pages (showing first 25 of {total})")
            print("\n📋 Available pages for your Q&A system:")
            
            valid_pages = []
            for i, page in enumerate(pages, 1):
                page_id = page.get('id')
                title = page.get('title', 'No title')
                page_type = page.get('type')
                version = page.get('version', {}).get('number', 'Unknown')
                
                # Create web URL
                web_ui = page.get('_links', {}).get('webui', '')
                page_url = f"{base_url}{web_ui}" if web_ui else f"{base_url}/pages/{page_id}"
                
                print(f"   {i:2d}. {title}")
                print(f"       ID: {page_id}")
                print(f"       Type: {page_type} | Version: {version}")
                print(f"       URL: {page_url}")
                
                valid_pages.append({
                    'id': page_id,
                    'title': title,
                    'space_key': space_key,
                    'url': page_url,
                    'version': version
                })
                print()
            
            # Test content retrieval for a few pages
            print("🧪 Test 3: Testing content retrieval...")
            for test_page in valid_pages[:3]:  # Test first 3 pages
                try:
                    page_id = test_page['id']
                    print(f"\n📖 Testing page: {test_page['title'][:40]}...")
                    
                    page_url = f"{base_url}/rest/api/content/{page_id}"
                    params = {'expand': 'body.storage,version,space'}
                    
                    page_response = session.get(page_url, params=params, timeout=10)
                    
                    if page_response.status_code == 200:
                        page_data = page_response.json()
                        body = page_data.get('body', {}).get('storage', {}).get('value', '')
                        print(f"   ✅ Content length: {len(body)} characters")
                        
                        if body:
                            # Clean HTML for preview
                            import re
                            clean_text = re.sub(r'<[^>]+>', '', body)
                            clean_text = ' '.join(clean_text.split())  # Normalize whitespace
                            preview = clean_text[:200] + "..." if len(clean_text) > 200 else clean_text
                            print(f"   📝 Preview: {preview}")
                        
                    else:
                        print(f"   ❌ Failed to get content: {page_response.status_code}")
                        
                except Exception as e:
                    print(f"   ❌ Error testing page {page_id}: {e}")
            
            # Generate configuration
            print("\n🔧 CONFIGURATION FOR YOUR SYSTEM:")
            print("-" * 50)
            print("Update your .env file:")
            print(f"CONFLUENCE_BASE_URL=https://incorta.atlassian.net/wiki")
            print(f"CONFLUENCE_USERNAME={username}")
            print(f"CONFLUENCE_API_TOKEN={api_token}")
            print(f"CONFLUENCE_SPACE_KEYS={space_key}")
            
            # Save test configuration
            config = {
                "base_url": f"{base_url}",
                "space_key": space_key,
                "space_name": pages[0].get('space', {}).get('name', space_key) if pages else space_key,
                "total_pages": total,
                "sample_pages": valid_pages[:10],  # First 10 pages for testing
                "working_page_ids": [p['id'] for p in valid_pages[:10]]
            }
            
            try:
                with open("incorta_space_config.json", "w") as f:
                    json.dump(config, f, indent=2)
                print(f"\n💾 Configuration saved to 'incorta_space_config.json'")
                print(f"   Found {len(valid_pages)} accessible pages")
                print(f"   Space key: {space_key}")
                
                # Suggest some good pages for testing
                if len(valid_pages) >= 3:
                    print(f"\n💡 Suggested test page IDs for your code:")
                    for i, page in enumerate(valid_pages[:5], 1):
                        print(f"   {i}. {page['id']} - {page['title'][:50]}")
                
            except Exception as e:
                print(f"❌ Could not save config: {e}")
            
            return True
            
        else:
            print(f"❌ Failed to get pages: {response.status_code}")
            if response.status_code == 403:
                print("   You may not have permission to list pages in this space")
                print("   But you can still access individual pages by ID")
            return False
            
    except Exception as e:
        print(f"❌ Error getting pages: {e}")
        return False

if __name__ == "__main__":
    # Update your .env first
    print("📝 Make sure your .env has:")
    print("CONFLUENCE_BASE_URL=https://incorta.atlassian.net/wiki")
    print("CONFLUENCE_USERNAME=mohamed.elsawy@incorta.com")
    print("CONFLUENCE_API_TOKEN=your_token_here")
    print()
    
    test_incorta_space_access()