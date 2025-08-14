import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from rag_pipeline import get_bot_response_with_context

# Load env variables
load_dotenv()
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")

# Initialize Slack Bolt app
bolt_app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
handler = SlackRequestHandler(bolt_app)

# Flask app
flask_app = Flask(__name__)


def get_thread_context(client, channel, thread_ts):
    """
    Fetch the thread conversation history to provide context
    """
    try:
        print(f"🧵 Fetching thread context for thread_ts: {thread_ts}")
        
        # Get thread replies
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
            
            # Get user info for human-readable names
            if user_id and not bot_id:
                try:
                    user_info = client.users_info(user=user_id)
                    username = user_info.get("user", {}).get("real_name") or user_info.get("user", {}).get("name", "User")
                except:
                    username = "User"
                
                thread_context.append(f"User ({username}): {text}")
            elif bot_id:
                # Clean bot responses (remove mentions, formatting)
                import re
                clean_text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
                clean_text = clean_text.replace("**", "")
                if clean_text:
                    thread_context.append(f"Bot: {clean_text}")
        
        context_text = "\n".join(thread_context)
        print(f"📝 Thread context extracted: {len(thread_context)} messages")
        return context_text
        
    except Exception as e:
        print(f"❌ Error fetching thread context: {str(e)}")
        return ""
    
# Add root route to handle the 404 error
@flask_app.route("/", methods=["GET", "POST"])
def root():
    if request.method == "POST":
        # Handle Slack URL verification
        data = request.get_json()
        if data and data.get("type") == "url_verification":
            return jsonify({"challenge": data.get("challenge")})
        # If it's not URL verification, redirect to slack events handler
        return handler.handle(request)
    return "Platform Knowledge Bot is running! 🤖"

# Respond to messages in channels with RAG integration
@bolt_app.event("message")
def handle_message_events(body, say, client):
    try:
        event = body.get("event", {})
        text = event.get("text", "")
        channel_type = event.get("channel_type", "")
        channel = event.get("channel", "")
        user = event.get("user", "")
        message_ts = event.get("ts", "")
        thread_ts = event.get("thread_ts")  # This exists if message is in a thread


        # Debug logging
        print(f"📩 MESSAGE EVENT: {event}")
        print(f"📝 Text: {text}")
        print(f"📱 Channel type: {channel_type}")
        print(f"👤 User: {user}")
        print(f"⏰ Message timestamp: {message_ts}")
        print(f"🧵 Thread timestamp: {thread_ts}")

        
        # Ignore bot messages
        if event.get("bot_id"):
            print("🤖 Ignoring bot message")
            return
            
        print(f"✅ Processing message: {text}")
        
        # Handle direct messages or check if bot is mentioned
        if channel_type == "im" or "<@" in text:
            if text.lower().__contains__("hello") or text.lower().__contains__("hi"):
                reply_thread_ts = thread_ts or message_ts
                say(text="Hi there! 👋 Ask me anything about the platform knowledge base!", 
                    thread_ts=reply_thread_ts)
            else:
                # Use RAG pipeline to get response
                print("🔍 Getting RAG response...")
                thread_context = ""
                if thread_ts:
                    print("🧵 Message is in a thread, getting context...")
                    thread_context = get_thread_context(client, channel, thread_ts)
                if thread_context:
                    response = get_bot_response_with_context(text, thread_context)
                else:
                    response = get_bot_response_with_context(text, "")
                print(f"💬 Sending response: {response[:100]}...")
                response = response.replace("**", "")
                say(text=response, thread_ts=message_ts)
        else:
            print("📵 Message not in DM or mention, ignoring")
        
    except Exception as e:
        print(f"❌ Error handling message: {str(e)}")
        say("Sorry, I encountered an error processing your message.")

# Handle app mentions
@bolt_app.event("app_mention")
def handle_app_mentions(body, say, client):
    try:
        event = body.get("event", {})
        text = event.get("text", "")
        user = event.get("user", "")
        channel = event.get("channel", "")
        message_ts = event.get("ts", "")
        thread_ts = event.get("thread_ts")

        
        # Debug logging
        print(f"🎯 APP MENTION EVENT: {event}")
        print(f"📝 Mention text: {text}")
        print(f"👤 User: {user}")
        print(f"⏰ Message timestamp: {message_ts}")
        print(f"🧵 Thread timestamp: {thread_ts}")
        
        # Remove bot mention from text
        import re
        clean_text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
        print(f"🧹 Clean text: {clean_text}")
        
        if clean_text:
            if clean_text.lower().__contains__("hello") or clean_text.lower().__contains__("hi"):
                say(text=f"<@{user}> Hi there! 👋 How can I assist you with the platform knowledge base?", 
                    thread_ts=message_ts)
            else:
                print("🔍 Getting RAG response for mention...")
                thread_context = ""
                if thread_ts:
                    print("🧵 Mention is in a thread, getting context...")
                    thread_context = get_thread_context(client, channel, thread_ts)
                
                # Get response with context
                if thread_context:
                    response = get_bot_response_with_context(clean_text, thread_context)
                else:
                    response = get_bot_response_with_context(clean_text, "")
                print(f"💬 Sending mention response: {response[:100]}...")
                response = response.replace("**", "")
                reply_thread_ts = thread_ts or message_ts
                say(text=f"<@{user}> {response}", thread_ts=reply_thread_ts)
        else:
            say(text=f"<@{user}> Hi! How can I help you with the platform knowledge base?", thread_ts=message_ts)

    except Exception as e:
        print(f"❌ Error handling mention: {str(e)}")
        say("Sorry, I encountered an error.")

# Health check
@flask_app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

# Slack events route
@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

if __name__ == "__main__":
    print("🚀 Starting Platform Knowledge Bot...")
    flask_app.run(host="0.0.0.0", port=3000, debug=True)