# confluence_webhook_handler_debug.py

from flask import Flask, request, jsonify
import json
import threading
from smart_qa_tracker import SmartQATracker
import os
from dotenv import load_dotenv
import traceback

load_dotenv()

class ConfluenceWebhookHandler:
    def __init__(self, smart_tracker: SmartQATracker):
        self.tracker = smart_tracker
        
    def handle_webhook(self, payload: dict):
        """Handle incoming webhook from Confluence Automation"""
        try:
            event_type = payload.get('eventType') or payload.get('event_type', '')
            print(f"📡 Received webhook event: {event_type}")
            
            if event_type == 'page_created':
                self.handle_page_created(payload)
            elif event_type == 'page_updated':
                self.handle_page_updated(payload)
            elif event_type == 'page_removed' or event_type == 'page_deleted':
                self.handle_page_removed(payload)
            else:
                print(f"⚠️ Unhandled event type: '{event_type}'")
                print(f"Available payload keys: {list(payload.keys())}")
                
        except Exception as e:
            print(f"❌ Error handling webhook: {e}")
            traceback.print_exc()
    
    def handle_page_created(self, payload: dict):
        """Handle page creation event from Confluence Automation"""
        try:
            # Support both formats
            if 'page_id' in payload:
                # Confluence Automation format
                page_id = payload.get('page_id')
                event_type = payload.get('event_type', 'page_created')
                print(f"📄 New page created (ID: {page_id}) - Automation format")
            else:
                # Standard webhook format
                page = payload.get('page', {})
                page_id = page.get('id')
                title = page.get('title', 'Unknown')
                print(f"📄 New page created: {title} (ID: {page_id}) - Standard format")
            
            if page_id:
                # Update the page Q&A in background (smart update)
                print(f"🚀 Starting Q&A update for page {page_id}")
                threading.Thread(
                    target=self.tracker.update_single_page_smart,
                    args=(page_id,)
                ).start()
            else:
                print("❌ No page_id found in payload")
                
        except Exception as e:
            print(f"Error handling page creation: {e}")
            traceback.print_exc()
    
    def handle_page_updated(self, payload: dict):
        """Handle page update event from Confluence Automation"""
        try:
            # Support both formats
            if 'page_id' in payload:
                # Confluence Automation format
                page_id = payload.get('page_id')
                event_type = payload.get('event_type', 'page_updated')
                print(f"📝 Page updated (ID: {page_id}) - Automation format")
            else:
                # Standard webhook format
                page = payload.get('page', {})
                page_id = page.get('id')
                title = page.get('title', 'Unknown')
                print(f"📝 Page updated: {title} (ID: {page_id}) - Standard format")
            
            if page_id:
                # Update the page Q&A in background (smart update)
                print(f"🚀 Starting Q&A update for page {page_id}")
                threading.Thread(
                    target=self.tracker.update_single_page_smart,
                    args=(page_id,)
                ).start()
            else:
                print("❌ No page_id found in payload")
                
        except Exception as e:
            print(f"Error handling page update: {e}")
            traceback.print_exc()
    
    def handle_page_removed(self, payload: dict):
        """Handle page removal event from Confluence Automation"""
        try:
            # Support both formats
            if 'page_id' in payload:
                # Confluence Automation format
                page_id = payload.get('page_id')
                event_type = payload.get('event_type', 'page_removed')
                print(f"🗑️ Page removed (ID: {page_id}) - Automation format")
            else:
                # Standard webhook format
                page = payload.get('page', {})
                page_id = page.get('id')
                title = page.get('title', 'Unknown')
                print(f"🗑️ Page removed: {title} (ID: {page_id}) - Standard format")
            
            if page_id:
                # Remove the page Q&A from vector store in background
                print(f"🗑️ Starting Q&A deletion for page {page_id}")
                threading.Thread(
                    target=self.tracker.delete_page_qa_pairs,
                    args=(page_id,)
                ).start()
            else:
                print("❌ No page_id found in payload")
                
        except Exception as e:
            print(f"Error handling page removal: {e}")
            traceback.print_exc()

# Flask app for webhook endpoint
webhook_app = Flask(__name__)

# Initialize Smart Q&A Tracker
smart_tracker = SmartQATracker(
    base_url=os.getenv("CONFLUENCE_BASE_URL"),
    username=os.getenv("CONFLUENCE_USERNAME"), 
    api_token=os.getenv("CONFLUENCE_API_TOKEN"),
    space_keys=os.getenv("CONFLUENCE_SPACE_KEYS", "").split(",") if os.getenv("CONFLUENCE_SPACE_KEYS") else None
)

webhook_handler = ConfluenceWebhookHandler(smart_tracker)

@webhook_app.route('/confluence/webhook', methods=['POST', 'GET'])
def confluence_webhook():
    """Endpoint to receive Confluence webhooks with enhanced debugging"""
    try:
        if request.method == 'GET':
            return jsonify({"message": "Confluence webhook endpoint is working", "method": "GET"}), 200
            
        print(f"\n🔍 DEBUG INFO:")
        print(f"Content-Type: {request.content_type}")
        print(f"Method: {request.method}")
        print(f"Headers: {dict(request.headers)}")
        
        # Get raw data first
        raw_data = request.get_data(as_text=True)
        print(f"Raw data length: {len(raw_data)}")
        print(f"Raw data preview: {raw_data[:500]}...")  # Show first 500 chars
        
        # Handle empty data
        if not raw_data or raw_data.strip() == '':
            print("⚠️ Empty request body - might be a webhook test")
            return jsonify({"status": "success", "message": "Empty webhook received - test OK"}), 200
        
        # Check content type
        content_type = request.content_type or ""
        if not content_type.startswith('application/json'):
            print(f"⚠️ Warning: Content-Type is '{content_type}', expected 'application/json'")
            
            # Try to handle form data or other formats
            if 'application/x-www-form-urlencoded' in content_type:
                print("🔄 Trying to parse as form data...")
                form_data = dict(request.form)
                print(f"Form data: {form_data}")
                return jsonify({"status": "success", "message": "Form data received", "data": form_data}), 200
        
        # Try to parse JSON with better error handling
        try:
            if raw_data:
                payload = json.loads(raw_data)
                print(f"✅ JSON parsed successfully")
                print(f"📦 Payload keys: {list(payload.keys()) if isinstance(payload, dict) else 'Not a dict'}")
            else:
                print("⚠️ No data received")
                return jsonify({"error": "No data received"}), 400
                
        except json.JSONDecodeError as json_error:
            print(f"❌ JSON Decode Error: {json_error}")
            print(f"📍 Error at position: {json_error.pos}")
            print(f"🔍 Characters around error:")
            if hasattr(json_error, 'pos') and json_error.pos > 0:
                start = max(0, json_error.pos - 50)
                end = min(len(raw_data), json_error.pos + 50)
                print(f"   '{raw_data[start:end]}'")
            
            # Try to fix common JSON issues
            fixed_data = raw_data.strip()
            # Remove any BOM or non-printable characters
            fixed_data = ''.join(char for char in fixed_data if ord(char) >= 32 or char in '\n\r\t')
            
            if fixed_data != raw_data:
                print("🔧 Attempting to fix JSON...")
                try:
                    payload = json.loads(fixed_data)
                    print("✅ JSON fixed and parsed successfully")
                except:
                    print("❌ JSON fix attempt failed")
                    return jsonify({
                        "error": "Invalid JSON format", 
                        "details": str(json_error),
                        "raw_data_preview": raw_data[:200]
                    }), 400
            else:
                return jsonify({
                    "error": "Invalid JSON format", 
                    "details": str(json_error),
                    "raw_data_preview": raw_data[:200]
                }), 400
        
        # Handle webhook in background
        print(f"🚀 Processing webhook...")
        threading.Thread(
            target=webhook_handler.handle_webhook,
            args=(payload,)
        ).start()
        
        return jsonify({"status": "success", "message": "Webhook processed"}), 200
        
    except Exception as e:
        print(f"❌ Error in webhook endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500

@webhook_app.route('/confluence/sync', methods=['POST'])
def manual_sync():
    """Endpoint to manually trigger a full sync"""
    try:
        print("🔄 Manual sync triggered")
        
        # Run smart sync in background
        threading.Thread(
            target=smart_tracker.sync_all_confluence_qa,
            kwargs={"force_regenerate": False}
        ).start()
        
        return jsonify({
            "status": "success", 
            "message": "Full sync started in background"
        }), 200
        
    except Exception as e:
        print(f"❌ Error in manual sync: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@webhook_app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Confluence Webhook Handler"
    }), 200

@webhook_app.route('/test', methods=['GET', 'POST'])
def test_endpoint():
    """Test endpoint to verify webhook setup"""
    if request.method == 'POST':
        return jsonify({
            "message": "POST test successful",
            "content_type": request.content_type,
            "data_received": bool(request.get_data())
        }), 200
    else:
        return jsonify({
            "message": "GET test successful",
            "webhook_url": "/confluence/webhook",
            "health_url": "/health"
        }), 200

if __name__ == "__main__":
    print("🚀 Starting Enhanced Confluence Webhook Handler with Debugging...")
    print("🔍 Debug mode enabled - detailed logging active")
    print("🧪 Test endpoint available at: /test")
    webhook_app.run(host="0.0.0.0", port=3001, debug=True)