# confluence_webhook_handler.py

from flask import Flask, request, jsonify
import json
import threading
from smart_qa_tracker import SmartQATracker
import os
from dotenv import load_dotenv

load_dotenv()

class ConfluenceWebhookHandler:
    def __init__(self, smart_tracker: SmartQATracker):
        self.tracker = smart_tracker
        
    def handle_webhook(self, payload: dict):
        """Handle incoming webhook from Confluence"""
        try:
            print(payload)
            event_type = payload.get('event_type', '')
            print(f"📡 Received webhook event: {event_type}")
            # print(f"Payload: {json.dumps(payload, indent=2)}")
            
            if event_type == 'page_created':
                self.handle_page_created(payload)
            elif event_type == 'page_updated':
                self.handle_page_updated(payload)
            elif event_type == 'page_removed':
                self.handle_page_removed(payload)
            elif event_type == 'page_trashed':
                self.handle_page_removed(payload)
            else:
                print(f"⚠️ Unhandled event type: {event_type}")
                
        except Exception as e:
            print(f"❌ Error handling webhook: {e}")
    
    def handle_page_created(self, payload: dict):
        """Handle page creation event"""
        try:
            page = payload.get('page', {})
            page_id = page.get('id')
            title = page.get('title', 'Unknown')
            
            print(f"📄 New page created: {title} (ID: {page_id})")
            
            # Update the page Q&A in background (smart update)
            threading.Thread(
                target=self.tracker.update_single_page_smart,
                args=(page_id,)
            ).start()
            
        except Exception as e:
            print(f"Error handling page creation: {e}")
    
    def handle_page_updated(self, payload: dict):
        """Handle page update event"""
        try:
            page = payload.get('page', {})
            page_id = page.get('id')
            title = page.get('title', 'Unknown')
            
            print(f"📝 Page updated: {title} (ID: {page_id})")
            
            # Update the page Q&A in background (smart update)
            threading.Thread(
                target=self.tracker.update_single_page_smart,
                args=(page_id,)
            ).start()
            
        except Exception as e:
            print(f"Error handling page update: {e}")
    
    def handle_page_removed(self, payload: dict):
        """Handle page removal event"""
        try:
            page = payload.get('page', {})
            page_id = page.get('id')
            title = page.get('title', 'Unknown')
            
            print(f"🗑️ Page removed: {title} (ID: {page_id})")
            
            # Remove the page Q&A from vector store in background
            threading.Thread(
                target=self.tracker.delete_page_qa_pairs,
                args=(page_id,)
            ).start()
            
        except Exception as e:
            print(f"Error handling page removal: {e}")

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

@webhook_app.route('/confluence/webhook', methods=['POST'])
def confluence_webhook():
    """Endpoint to receive Confluence webhooks"""
    try:
        # Verify content type
        if request.content_type != 'application/json':
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "No JSON payload received"}), 400
        
        print(f"📡 Webhook received: {json.dumps(payload, indent=2)}")
        print(f"Event type: {payload.get('eventType', 'Unknown')}")
        
        # Handle webhook in background
        threading.Thread(
            target=webhook_handler.handle_webhook,
            args=(payload,)
        ).start()
        
        return jsonify({"status": "success", "message": "Webhook processed"}), 200
        
    except Exception as e:
        print(f"❌ Error in webhook endpoint: {e}")
        return jsonify({"error": str(e)}), 500

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
        return jsonify({"error": str(e)}), 500

@webhook_app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Confluence Webhook Handler"
    }), 200

if __name__ == "__main__":
    print("🚀 Starting Confluence Webhook Handler...")
    webhook_app.run(host="0.0.0.0", port=3001, debug=True)