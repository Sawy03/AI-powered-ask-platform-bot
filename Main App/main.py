#!/usr/bin/env python3
"""
Production Main Entry Point for Platform Knowledge Bot
Combines Slack Bot, Confluence Webhook Handler, and QA RAG Pipeline
Uses HTTP mode for Slack events
"""

import os
import json
import threading
import traceback
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
import re
import time

# Import our custom modules
from qa_rag_pipeline import (
    get_bot_response_with_context, 
    initialize_confident_qa_vector_store,
    initialize_confluence_qa_data
)
from smart_qa_tracker import SmartQATracker

# Load environment variables
load_dotenv()

# Environment variables
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_SPACE_KEYS = os.getenv("CONFLUENCE_SPACE_KEYS", "").split(",") if os.getenv("CONFLUENCE_SPACE_KEYS") else None

# Initialize Flask app
app = Flask(__name__)

# Initialize Slack Bolt app (HTTP mode only)
bolt_app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
slack_handler = SlackRequestHandler(bolt_app)

# Initialize Smart Q&A Tracker (shared instance)
smart_tracker = SmartQATracker(
    base_url=CONFLUENCE_BASE_URL,
    username=CONFLUENCE_USERNAME,
    api_token=CONFLUENCE_API_TOKEN,
    space_keys=CONFLUENCE_SPACE_KEYS
)

# ============================================================================
# CONFLUENCE WEBHOOK HANDLER FUNCTIONALITY
# ============================================================================

class ConfluenceWebhookHandler:
    def __init__(self, smart_tracker: SmartQATracker):
        self.tracker = smart_tracker
        
    def handle_webhook(self, payload: dict):
        """Handle incoming webhook from Confluence Automation"""
        try:
            event_type = payload.get('eventType') or payload.get('event_type', '')
            print(f"Received webhook event: {event_type}")
            
            if event_type == 'page_created':
                self.handle_page_created(payload)
            elif event_type == 'page_updated':
                self.handle_page_updated(payload)
            elif event_type == 'page_removed' or event_type == 'page_deleted':
                self.handle_page_removed(payload)
            else:
                print(f"Unhandled event type: '{event_type}'")
                print(f"Available payload keys: {list(payload.keys())}")
                
        except Exception as e:
            print(f"Error handling webhook: {e}")
            traceback.print_exc()
    
    def handle_page_created(self, payload: dict):
        """Handle page creation event from Confluence Automation"""
        try:
            if 'page_id' in payload:
                page_id = payload.get('page_id')
                print(f"New page created (ID: {page_id}) - Automation format")
            else:
                page = payload.get('page', {})
                page_id = page.get('id')
                title = page.get('title', 'Unknown')
                print(f"New page created: {title} (ID: {page_id}) - Standard format")
            
            if page_id:
                print(f"Starting Q&A update for page {page_id}")
                threading.Thread(
                    target=self.tracker.update_single_page_smart,
                    args=(page_id,)
                ).start()
            else:
                print("No page_id found in payload")
                
        except Exception as e:
            print(f"Error handling page creation: {e}")
            traceback.print_exc()
    
    def handle_page_updated(self, payload: dict):
        """Handle page update event from Confluence Automation"""
        try:
            if 'page_id' in payload:
                page_id = payload.get('page_id')
                print(f"Page updated (ID: {page_id}) - Automation format")
            else:
                page = payload.get('page', {})
                page_id = page.get('id')
                title = page.get('title', 'Unknown')
                print(f"Page updated: {title} (ID: {page_id}) - Standard format")
            
            if page_id:
                print(f"Starting Q&A update for page {page_id}")
                threading.Thread(
                    target=self.tracker.update_single_page_smart,
                    args=(page_id,)
                ).start()
            else:
                print("No page_id found in payload")
                
        except Exception as e:
            print(f"Error handling page update: {e}")
            traceback.print_exc()
    
    def handle_page_removed(self, payload: dict):
        """Handle page removal event from Confluence Automation"""
        try:
            if 'page_id' in payload:
                page_id = payload.get('page_id')
                print(f"Page removed (ID: {page_id}) - Automation format")
            else:
                page = payload.get('page', {})
                page_id = page.get('id')
                title = page.get('title', 'Unknown')
                print(f"Page removed: {title} (ID: {page_id}) - Standard format")
            
            if page_id:
                print(f"Starting Q&A deletion for page {page_id}")
                threading.Thread(
                    target=self.tracker.delete_page_qa_pairs,
                    args=(page_id,)
                ).start()
            else:
                print("No page_id found in payload")
                
        except Exception as e:
            print(f"Error handling page removal: {e}")
            traceback.print_exc()

# Initialize webhook handler
webhook_handler = ConfluenceWebhookHandler(smart_tracker)

# ============================================================================
# SLACK HELPER FUNCTIONS
# ============================================================================

def get_thread_context(client, channel, thread_ts):
    """Fetch the thread conversation history to provide context"""
    try:
        print(f"Fetching thread context for thread_ts: {thread_ts}")
        
        result = client.conversations_replies(
            channel=channel,
            ts=thread_ts,
            inclusive=True
        )
        
        messages = result.get("messages", [])
        thread_context = []
        
        for msg in messages:
            user_id = msg.get("user")
            text = msg.get("text", "")
            bot_id = msg.get("bot_id")
            
            if user_id and not bot_id:
                try:
                    user_info = client.users_info(user=user_id)
                    username = user_info.get("user", {}).get("real_name") or user_info.get("user", {}).get("name", "User")
                except:
                    username = "User"
                
                thread_context.append(f"User ({username}): {text}")
            elif bot_id:
                clean_text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
                clean_text = clean_text.replace("**", "")
                if clean_text:
                    thread_context.append(f"Bot: {clean_text}")
        
        context_text = "\n".join(thread_context)
        print(f"Thread context extracted: {len(thread_context)} messages")
        return context_text
        
    except Exception as e:
        print(f"Error fetching thread context: {str(e)}")
        return ""

def get_parent_message(client, channel, thread_ts):
    """Fetch only the parent message of a thread to provide as context"""
    try:
        print(f"Fetching parent message for thread_ts: {thread_ts}")
        
        result = client.conversations_history(
            channel=channel,
            latest=thread_ts,
            inclusive=True,
            limit=1
        )
        
        messages = result.get("messages", [])
        if not messages:
            print("No parent message found.")
            return ""

        msg = messages[0]
        user_id = msg.get("user")
        text = msg.get("text", "")
        bot_id = msg.get("bot_id")
        
        if user_id and not bot_id:
            try:
                user_info = client.users_info(user=user_id)
                username = user_info.get("user", {}).get("real_name") or user_info.get("user", {}).get("name", "User")
            except:
                username = "User"
            
            context_text = text
        elif bot_id:
            clean_text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
            clean_text = clean_text.replace("**", "")
            if clean_text:
                context_text = clean_text
            else:
                context_text = ""
        else:
            context_text = ""

        print(f"Parent message extracted: {context_text}")
        return context_text
        
    except Exception as e:
        print(f"Error fetching parent message: {str(e)}")
        return ""

# ============================================================================
# SLACK EVENT HANDLERS
# ============================================================================

@bolt_app.event("message")
def handle_message_events(body, say, client):
    """Handle direct messages and mentions"""
    try:
        event = body.get("event", {})
        text = event.get("text", "")
        channel_type = event.get("channel_type", "")
        channel = event.get("channel", "")
        user = event.get("user", "")
        message_ts = event.get("ts", "")
        thread_ts = event.get("thread_ts")

        print(f"MESSAGE EVENT: {event}")
        print(f"Text: {text}")
        print(f"Channel type: {channel_type}")
        print(f"User: {user}")
        print(f"Message timestamp: {message_ts}")
        print(f"Thread timestamp: {thread_ts}")

        # Ignore bot messages
        if event.get("bot_id"):
            print("Ignoring bot message")
            return
            
        print(f"Processing message: {text}")
        
        # Handle direct messages or check if bot is mentioned
        if channel_type == "im" or "<@" in text:
            if text.lower().split(" ").__contains__("hello") or text.lower().split(" ").__contains__("hi"):
                reply_thread_ts = thread_ts or message_ts
                say(text="Hi there! Ask me anything about the platform knowledge base!", 
                    thread_ts=reply_thread_ts)
            elif "correction" in text.lower():
                parent_message = ""
                if thread_ts:
                    parent_message = get_parent_message(client, channel, thread_ts)
                reply_thread_ts = thread_ts or message_ts
                print(parent_message)
                print(text)
                sawy = text.lower().replace("correction", "")
                print(sawy)
                smart_tracker.save_confident_answer(parent_message, sawy)
                say(text="This question has been sent for correction!", 
                    thread_ts=reply_thread_ts)
            else:
                print("Getting RAG response...")
                thread_context = ""
                if thread_ts:
                    print("Message is in a thread, getting context...")
                    thread_context = get_thread_context(client, channel, thread_ts)
                if thread_context:
                    response = get_bot_response_with_context(text, thread_context)
                else:
                    response = get_bot_response_with_context(text, "")
                print(f"Sending response: {response[:100]}...")
                response = response.replace("**", "")
                say(text=response, thread_ts=message_ts)
        else:
            print("Message not in DM or mention, ignoring")
        
    except Exception as e:
        print(f"Error handling message: {str(e)}")
        say("Sorry, I encountered an error processing your message.")

@bolt_app.event("app_mention")
def handle_app_mentions(body, say, client):
    """Handle app mentions in channels"""
    try:
        event = body.get("event", {})
        text = event.get("text", "")
        user = event.get("user", "")
        channel = event.get("channel", "")
        message_ts = event.get("ts", "")
        thread_ts = event.get("thread_ts")

        print(f"APP MENTION EVENT: {event}")
        print(f"Mention text: {text}")
        print(f"User: {user}")
        print(f"Message timestamp: {message_ts}")
        print(f"Thread timestamp: {thread_ts}")
        
        # Remove bot mention from text
        clean_text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
        print(f"Clean text: {clean_text}")
        
        if clean_text:
            if "hello" in clean_text.lower() or "hi" in clean_text.lower():
                say(text=f"<@{user}> Hi there! How can I assist you with the platform knowledge base?", 
                    thread_ts=message_ts)
            elif "correction" in clean_text.lower():
                parent_message = ""
                if thread_ts:
                    parent_message = get_parent_message(client, channel, thread_ts)
                reply_thread_ts = thread_ts or message_ts
                print(parent_message)
                print(clean_text)
                sawy2 = parent_message.replace("<@U099VBD9BR7>", "")
                sawy = clean_text.lower().replace("correction", "")
                print(sawy2)
                print(sawy)
                smart_tracker.save_confident_answer(sawy2, sawy)
                say(text=f"<@{user}> This question has been sent for correction!", 
                    thread_ts=reply_thread_ts)
            else:
                print("Getting RAG response for mention...")
                thread_context = ""
                if thread_ts:
                    print("Mention is in a thread, getting context...")
                    thread_context = get_thread_context(client, channel, thread_ts)
                
                if thread_context:
                    response = get_bot_response_with_context(clean_text, thread_context)
                else:
                    response = get_bot_response_with_context(clean_text, "")
                print(f"Sending mention response: {response[:100]}...")
                response = response.replace("**", "")
                reply_thread_ts = thread_ts or message_ts
                say(text=f"<@{user}> {response}", thread_ts=reply_thread_ts)
        else:
            say(text=f"<@{user}> Hi! How can I help you with the platform knowledge base?", thread_ts=message_ts)

    except Exception as e:
        print(f"Error handling mention: {str(e)}")
        say("Sorry, I encountered an error.")

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route("/", methods=["GET", "POST"])
def root():
    """Root endpoint"""
    if request.method == "POST":
        # Handle Slack URL verification
        data = request.get_json()
        if data and data.get("type") == "url_verification":
            return jsonify({"challenge": data.get("challenge")})
        # If it's not URL verification, redirect to slack events handler
        return slack_handler.handle(request)
    return "Platform Knowledge Bot is running!"

@app.route("/slack/events", methods=["POST"])
def slack_events():
    """Slack events endpoint"""
    print(f"Received Slack event: {request.get_json()}")
    return slack_handler.handle(request)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Platform Knowledge Bot",
        "components": {
            "slack_bot": "active",
            "confluence_webhook": "active", 
            "qa_pipeline": "active"
        }
    }), 200

# ============================================================================
# CONFLUENCE WEBHOOK ROUTES
# ============================================================================

@app.route('/confluence/webhook', methods=['POST', 'GET'])
def confluence_webhook():
    """Endpoint to receive Confluence webhooks with enhanced debugging"""
    try:
        if request.method == 'GET':
            return jsonify({"message": "Confluence webhook endpoint is working", "method": "GET"}), 200
            
        print(f"\nDEBUG INFO:")
        print(f"Content-Type: {request.content_type}")
        print(f"Method: {request.method}")
        print(f"Headers: {dict(request.headers)}")
        
        # Get raw data first
        raw_data = request.get_data(as_text=True)
        print(f"Raw data length: {len(raw_data)}")
        print(f"Raw data preview: {raw_data[:500]}...")
        
        # Handle empty data
        if not raw_data or raw_data.strip() == '':
            print("Empty request body - might be a webhook test")
            return jsonify({"status": "success", "message": "Empty webhook received - test OK"}), 200
        
        # Check content type
        content_type = request.content_type or ""
        if not content_type.startswith('application/json'):
            print(f"Warning: Content-Type is '{content_type}', expected 'application/json'")
            
            if 'application/x-www-form-urlencoded' in content_type:
                print("Trying to parse as form data...")
                form_data = dict(request.form)
                print(f"Form data: {form_data}")
                return jsonify({"status": "success", "message": "Form data received", "data": form_data}), 200
        
        # Try to parse JSON with better error handling
        try:
            if raw_data:
                payload = json.loads(raw_data)
                print(f"JSON parsed successfully")
                print(f"Payload keys: {list(payload.keys()) if isinstance(payload, dict) else 'Not a dict'}")
            else:
                print("No data received")
                return jsonify({"error": "No data received"}), 400
                
        except json.JSONDecodeError as json_error:
            print(f"JSON Decode Error: {json_error}")
            print(f"Error at position: {json_error.pos}")
            print(f"Characters around error:")
            if hasattr(json_error, 'pos') and json_error.pos > 0:
                start = max(0, json_error.pos - 50)
                end = min(len(raw_data), json_error.pos + 50)
                print(f"   '{raw_data[start:end]}'")
            
            # Try to fix common JSON issues
            fixed_data = raw_data.strip()
            fixed_data = ''.join(char for char in fixed_data if ord(char) >= 32 or char in '\n\r\t')
            
            if fixed_data != raw_data:
                print("Attempting to fix JSON...")
                try:
                    payload = json.loads(fixed_data)
                    print("JSON fixed and parsed successfully")
                except:
                    print("JSON fix attempt failed")
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
        print(f"Processing webhook...")
        threading.Thread(
            target=webhook_handler.handle_webhook,
            args=(payload,)
        ).start()
        
        return jsonify({"status": "success", "message": "Webhook processed"}), 200
        
    except Exception as e:
        print(f"Error in webhook endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500

@app.route('/confluence/sync', methods=['POST'])
def manual_sync():
    """Endpoint to manually trigger a full sync"""
    try:
        print("Manual sync triggered")
        
        threading.Thread(
            target=smart_tracker.sync_all_confluence_qa,
            kwargs={"force_regenerate": False}
        ).start()
        
        return jsonify({
            "status": "success", 
            "message": "Full sync started in background"
        }), 200
        
    except Exception as e:
        print(f"Error in manual sync: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ============================================================================
# QA MANAGEMENT ROUTES
# ============================================================================

@app.route('/qa/confident', methods=['GET'])
def get_confident_qa():
    """Endpoint to retrieve all confident Q&A pairs"""
    try:
        confident_qa = smart_tracker.get_confident_qa_pairs()
        print(f"Retrieved {len(confident_qa)} confident Q&A pairs.")
        return jsonify(confident_qa), 200
    except Exception as e:
        print(f"Error retrieving confident Q&A pairs: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/qa/general', methods=['GET'])
def get_general_qa():
    """Endpoint to retrieve all general Q&A pairs from the knowledge base"""
    try:
        general_qa = smart_tracker.get_general_qa_pairs()
        print(f"Retrieved {len(general_qa)} general Q&A pairs.")
        return jsonify(general_qa), 200
    except Exception as e:
        print(f"Error retrieving general Q&A pairs: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/qa/confident/delete/<int:pair_id>', methods=['DELETE'])
def delete_confident_qa_pair(pair_id):
    """Endpoint to delete a single Q&A pair by ID"""
    try:
        deleted_count = smart_tracker.delete_confident_qa_pair_by_id(pair_id)
        print("\n1. Cleaning database of invalid entries...")
        smart_tracker.clean_confident_database()
    
        print("\n2. Recreating confident vector store...")
        smart_tracker.recreate_confident_vector_store()
        if deleted_count > 0:
            print(f"Successfully deleted Q&A pair with ID: {pair_id}")
            return jsonify({"status": "success", "message": f"Q&A pair with ID {pair_id} deleted."}), 200
        else:
            print(f"Q&A pair with ID {pair_id} not found.")
            return jsonify({"status": "error", "message": f"Q&A pair with ID {pair_id} not found."}), 404
    except Exception as e:
        print(f"Error deleting Q&A pair: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET', 'POST'])
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

# ============================================================================
# INITIALIZATION AND STARTUP
# ============================================================================

def initialize_system():
    """Initialize the system components"""
    print("Initializing Platform Knowledge Bot...")
    
    try:
        print("1. Initializing confident Q&A vector store...")
        initialize_confident_qa_vector_store()
        
        print("2. Initializing confluence Q&A data...")
        initialize_confluence_qa_data(force_regenerate=False)
        
        print("System initialization completed successfully!")
        
    except Exception as e:
        print(f"Error during system initialization: {e}")
        traceback.print_exc()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("Starting Platform Knowledge Bot (Production Mode)...")
    print("Components: Slack Bot + Confluence Webhook Handler + QA RAG Pipeline")
    print("Running in HTTP mode (no Socket Mode)")
    
    # Initialize system in background
    initialization_thread = threading.Thread(target=initialize_system)
    initialization_thread.daemon = True
    initialization_thread.start()
    
    # Start Flask app on port 3000
    app.run(host="0.0.0.0", port=3000, debug=False)