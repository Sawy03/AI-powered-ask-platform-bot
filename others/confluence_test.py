#!/usr/bin/env python3
"""
Test Confluence API connection with multiple endpoint attempts
"""
import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()

def test_confluence_connection():
    """Test basic Confluence API connection"""
    
    base_url = os.getenv("CONFLUENCE_BASE_URL")
    username = os.getenv("CONFLUENCE_USERNAME") 
    api_token = os.getenv("CONFLUENCE_API_TOKEN")
    
    print("🔧 CONFLUENCE CONNECTION TEST")
    print("=" * 50)
    print(f"Base URL: {base_url}")
    print(f"Username: {username}")
    print(f"API Token: {'*' * (len(api_token)-8) + api_token[-8:] if api_token else 'Not set'}")
    print()
    
    if not all([base_url, username, api_token]):
        print("❌ Missing configuration! Check your .env file")
        return False
    
    # Clean API token (remove quotes if present)
    api_token = api_token.strip('"\'')
    
    session = requests.Session()
    session.auth = HTTPBasicAuth(username, api_token)
    
    # Test different base URLs and API endpoints
    test_urls = [
        # Original URL with different API paths
        {
            'base': base_url,
            'endpoints': [
                '/rest/api/space',
                '/wiki/rest/api/space', 
                '/confluence/rest/api/space',
                '/rest/api/content',
                '/wiki/rest/api/content',
                '/confluence/rest/api/content'
            ]
        },
        # Try with /wiki prefix
        {
            'base': base_url.rstrip('/') + '/wiki',
            'endpoints': [
                '/rest/api/space',
                '/rest/api/content'
            ]
        },
        # Try with /confluence prefix  
        {
            'base': base_url.rstrip('/') + '/confluence',
            'endpoints': [
                '/rest/api/space',
                '/rest/api/content'
            ]
        }
    ]
    
    successful_config = None
    
    print("🔍 Testing different API endpoints...")
    print("-" * 50)
    
    for config in test_urls:
        base = config['base']
        print(f"\n🧪 Testing base URL: {base}")
        
        for endpoint in config['endpoints']:
            try:
                full_url = f"{base}{endpoint}"
                print(f"  📡 Trying: {full_url}")
                
                response = session.get(full_url, params={'limit': 3}, timeout=10)
                print(f"    Status: {response.status_code}")
                
                if response.status_code == 200:
                    print(f"    ✅ SUCCESS! Working endpoint found")
                    successful_config = {'base': base, 'endpoint': endpoint}
                    
                    # Parse response
                    try:
                        data = response.json()
                        if 'results' in data:
                            results = data.get('results', [])
                            print(f"    📊 Found {len(results)} items")
                            
                            if endpoint.endswith('/space'):
                                print(f"    📁 Spaces:")
                                for item in results[:3]:
                                    print(f"      - {item.get('name')} ({item.get('key')})")
                            else:
                                print(f"    📄 Pages:")
                                for item in results[:3]:
                                    print(f"      - {item.get('title', 'No title')[:40]} (ID: {item.get('id')})")
                        else:
                            print(f"    📦 Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                            
                    except Exception as parse_e:
                        print(f"    ⚠️ Could not parse JSON: {parse_e}")
                    
                    break  # Found working endpoint, stop testing for this base
                    
                elif response.status_code == 401:
                    print(f"    🔐 Authentication failed")
                elif response.status_code == 403:
                    print(f"    🚫 Access forbidden")
                elif response.status_code == 404:
                    print(f"    ❌ Endpoint not found")
                else:
                    print(f"    ⚠️ Status {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"    ⏱️ Timeout")
            except requests.exceptions.ConnectionError:
                print(f"    🔌 Connection error")
            except Exception as e:
                print(f"    ❌ Error: {str(e)[:50]}")
        
        if successful_config:
            break  # Found working config, stop testing
    
    if not successful_config:
        print("\n❌ No working API endpoint found!")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check if this is actually a Confluence instance (not just Jira)")
        print("2. Verify the base URL is correct")
        print("3. Make sure you have Confluence access (not just Jira)")
        print("4. Check your API token permissions")
        print("5. Try accessing Confluence in your browser first")
        return False
    
    print(f"\n🎉 SUCCESS! Working configuration found:")
    print(f"   Base URL: {successful_config['base']}")
    print(f"   Endpoint: {successful_config['endpoint']}")
    
    # Now test the specific page
    if successful_config:
        print(f"\n🧪 Testing specific page access (ID: 1763213371)")
        
        page_endpoints = [
            '/rest/api/content/1763213371',
            '/rest/api/content/1763213371?expand=body.storage,version,space'
        ]
        
        for endpoint in page_endpoints:
            try:
                full_url = f"{successful_config['base']}{endpoint}"
                print(f"  📡 Trying: {full_url}")
                
                response = session.get(full_url, timeout=10)
                print(f"    Status: {response.status_code}")
                
                if response.status_code == 200:
                    page_data = response.json()
                    print(f"    ✅ Page found!")
                    print(f"      Title: {page_data.get('title')}")
                    print(f"      Space: {page_data.get('space', {}).get('name', 'Unknown')}")
                    print(f"      Type: {page_data.get('type')}")
                    return True
                elif response.status_code == 404:
                    print(f"    ❌ Page not found")
                elif response.status_code == 403:
                    print(f"    🚫 Access forbidden to page")
                else:
                    print(f"    ⚠️ Status {response.status_code}")
                    
            except Exception as e:
                print(f"    ❌ Error: {str(e)[:50]}")
    
    # Update .env recommendation
    print(f"\n📝 Update your .env file with the working configuration:")
    print(f"CONFLUENCE_BASE_URL={successful_config['base']}")
    
    return True

if __name__ == "__main__":
    test_confluence_connection()