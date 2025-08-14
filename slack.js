const { App } = require('@slack/bolt');
const express = require('express');
require('dotenv').config();

// You'll need to create this file based on your RAG pipeline
const { getBotResponse } = require('./rag_pipeline');


// Initialize Slack Bolt app
const bolt_app = new App({
    token: SLACK_BOT_TOKEN,
    signingSecret: SLACK_SIGNING_SECRET,
    socketMode: false,
    appToken: process.env.SLACK_APP_TOKEN // Optional for Socket Mode
});

// Express app
const express_app = express();

// Middleware
express_app.use(express.json());
express_app.use(express.urlencoded({ extended: true }));

// Root route
express_app.get('/', (req, res) => {
    res.send('Platform Knowledge Bot is running! 🤖');
});

express_app.post('/', async (req, res) => {
    // Handle Slack URL verification
    const { type, challenge } = req.body;
    if (type === 'url_verification') {
        return res.json({ challenge });
    }
    
    // Pass to Slack handler
    return bolt_app.receiver.handle(req, res);
});

// Handle message events
bolt_app.event('message', async ({ event, say }) => {
    try {
        const { text = '', channel_type = '', user = '', bot_id } = event;
        
        // Debug logging
        console.log('📩 MESSAGE EVENT:', event);
        console.log('📝 Text:', text);
        console.log('📱 Channel type:', channel_type);
        console.log('👤 User:', user);
        
        // Ignore bot messages
        if (bot_id) {
            console.log('🤖 Ignoring bot message');
            return;
        }
        
        console.log('✅ Processing message:', text);
        
        // Handle direct messages or mentions
        if (channel_type === 'im' || text.includes('<@')) {
            const lowerText = text.toLowerCase();
            if (lowerText.includes('hello') || lowerText.includes('hi')) {
                await say('Hi there! 👋 Ask me anything about the platform knowledge base!');
            } else {
                // Use RAG pipeline to get response
                console.log('🔍 Getting RAG response...');
                try {
                    const response = await getBotResponse(text);
                    console.log('💬 Sending response:', response.substring(0, 100) + '...');
                    await say(response);
                } catch (error) {
                    console.error('❌ Error getting bot response:', error);
                    await say('Sorry, I encountered an error processing your message.');
                }
            }
        } else {
            console.log('📵 Message not in DM or mention, ignoring');
        }
        
    } catch (error) {
        console.error('❌ Error handling message:', error);
        await say('Sorry, I encountered an error processing your message.');
    }
});

// Handle app mentions
bolt_app.event('app_mention', async ({ event, say }) => {
    try {
        const { text = '', user = '' } = event;
        
        // Debug logging
        console.log('🎯 APP MENTION EVENT:', event);
        console.log('📝 Mention text:', text);
        console.log('👤 User:', user);
        
        // Remove bot mention from text
        const cleanText = text.replace(/<@[A-Z0-9]+>/g, '').trim();
        console.log('🧹 Clean text:', cleanText);
        
        if (cleanText) {
            const lowerText = cleanText.toLowerCase();
            if (lowerText.includes('hello') || lowerText.includes('hi')) {
                await say('Hi there! 👋 How can I assist you with the platform knowledge base?');
            } else {
                console.log('🔍 Getting RAG response for mention...');
                try {
                    const response = await getBotResponse(cleanText);
                    console.log('💬 Sending mention response:', response.substring(0, 100) + '...');
                    const cleanResponse = response.replace(/\*\*/g, '');
                    await say(`<@${user}> ${cleanResponse}`);
                } catch (error) {
                    console.error('❌ Error getting bot response:', error);
                    await say(`<@${user}> Sorry, I encountered an error processing your request.`);
                }
            }
        } else {
            await say(`<@${user}> Hi! How can I help you with the platform knowledge base?`);
        }
        
    } catch (error) {
        console.error('❌ Error handling mention:', error);
        await say('Sorry, I encountered an error.');
    }
});

// Health check
express_app.get('/health', (req, res) => {
    res.json({ status: 'healthy' });
});

// Slack events route
express_app.post('/slack/events', (req, res) => {
    return bolt_app.receiver.handle(req, res);
});

// Start the server
const PORT = process.env.PORT || 3000;

async function startServer() {
    try {
        await bolt_app.start();
        console.log('⚡️ Bolt app is running!');
        
        express_app.listen(PORT, '0.0.0.0', () => {
            console.log(`🚀 Platform Knowledge Bot is running on port ${PORT}!`);
        });
    } catch (error) {
        console.error('❌ Error starting server:', error);
    }
}

startServer();