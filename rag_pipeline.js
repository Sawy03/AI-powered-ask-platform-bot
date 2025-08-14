const SimpleVectorStore = require('./vector2');
const axios = require('axios');

class RAGPipeline {
    constructor() {
        this.vectorStore = new SimpleVectorStore();
        this.initialized = false;
        
        // Ollama configuration (matching your Python setup)
        this.aiModelUrl = 'http://127.0.0.1:11434/api/generate';
        this.modelName = 'llama3.2:1b'; // Match your Python model
    }

    async initialize() {
        if (!this.initialized) {
            await this.vectorStore.initialize();
            this.initialized = true;
        }
    }

    async callLocalAI(prompt) {
        try {
            console.log('🤖 Calling local AI model...');
            
            const response = await axios.post(this.aiModelUrl, {
                model: this.modelName,
                prompt: prompt,
                stream: false,
                options: {
                    temperature: 0.7,
                    num_predict: 500
                }
            }, {
                timeout: 60000,
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.data && response.data.response) {
                return response.data.response.trim();
            } else {
                throw new Error('Invalid response format from AI model');
            }

        } catch (error) {
            console.error('❌ Error calling local AI model:', error.message);
            throw new Error(`Unable to connect to local AI model: ${error.message}`);
        }
    }

    createPrompt(userQuestion, contextQA) {
        // Create context from Q&A pairs (matching Python logic exactly)
        let context = "";
        
        contextQA.forEach((result, index) => {
            const metadata = result.metadata;
            const question = metadata.question;
            const answer = metadata.answer;
            const source = metadata.source;
            // Format exactly like Python version
            context += `Source: ${source}\nQ: ${question}\n\nA: ${answer}\n\n`;
        });

        // Use the EXACT same prompt template as your Python version
        const prompt = `You are a helpful AI assistant for the platform team's knowledge base. Answer questions based ONLY on the provided context.

Be concise and helpful. If you don't have enough information, say so clearly.

Context from knowledge base:
${context}

Question: ${userQuestion}

Answer:`;

        return prompt;
    }

    async getBotResponse(query) {
        try {
            await this.initialize();
            
            console.log(`🔍 Searching for relevant questions for: "${query}"`);
            
            // Use the question search method (k=5, matching Python)
            const relevantQuestions = await this.vectorStore.questionSearch(query, 5, 0.3);
            
            if (relevantQuestions.length === 0) {
                return "Sorry, I couldn't find relevant information in the knowledge base for your question. Please contact the platform team directly.";
            }

            console.log(`📚 Found ${relevantQuestions.length} relevant questions`);
            
            // Create prompt with context
            const prompt = this.createPrompt(query, relevantQuestions);
            
            // ALWAYS try to call the AI model first
            console.log('🤖 Generating response with local AI model...');
            
            try {
                const aiResponse = await this.callLocalAI(prompt);
                
                // Get unique sources (matching Python logic)
                const sources = relevantQuestions.slice(0, 3).map(result => result.metadata.source);
                const uniqueSources = [...new Set(sources)];
                
                let sourcesInfo = "";
                if (uniqueSources.length > 0) {
                    sourcesInfo = `\n\n📚 Sources: ${uniqueSources.slice(0, 2).join(', ')}`;
                }
                
                const fullResponse = aiResponse + sourcesInfo;
                
                console.log('✅ AI response generated successfully');
                return fullResponse;
                
            } catch (aiError) {
                console.error('❌ AI model error:', aiError.message);
                console.log('📋 Falling back to direct context response');
                
                // Fallback: provide a structured answer from the context
                const bestMatch = relevantQuestions[0];
                const metadata = bestMatch.metadata;
                
                let fallbackResponse = `Based on the knowledge base:\n\n${metadata.answer}`;
                
                if (metadata.source) {
                    fallbackResponse += `\n\n📚 Source: ${metadata.source}`;
                }
                
                return fallbackResponse;
            }

        } catch (error) {
            console.error('❌ Error in RAG pipeline:', error);
            return `Sorry, I encountered an error: ${error.message}`;
        }
    }

    // Check if the AI model is available
    async checkModelStatus() {
        try {
            const response = await axios.get('http://localhost:11434/api/tags', {
                timeout: 5000
            });
            
            const models = response.data.models || [];
            const hasModel = models.some(model => model.name.includes(this.modelName));
            
            if (hasModel) {
                console.log(`✅ AI model ${this.modelName} is available`);
                return true;
            } else {
                console.log(`❌ AI model ${this.modelName} not found. Available models:`, models.map(m => m.name));
                return false;
            }
            
        } catch (error) {
            console.log(`❌ Cannot connect to Ollama server: ${error.message}`);
            return false;
        }
    }
}

// Create singleton instance
const ragPipeline = new RAGPipeline();

async function getBotResponse(query) {
    try {
        // Check model status first
        const modelAvailable = await ragPipeline.checkModelStatus();
        if (!modelAvailable) {
            console.log('⚠️  AI model not available, will use fallback in getBotResponse');
        }
        
        return await ragPipeline.getBotResponse(query);
    } catch (error) {
        console.error('❌ Error in getBotResponse:', error);
        return `Sorry, I encountered an error: ${error.message}`;
    }
}

module.exports = { getBotResponse };