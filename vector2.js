const fs = require('fs');
const csv = require('csv-parser');

class SimpleVectorStore {
    constructor() {
        this.documents = [];
        this.csvFile = "ai_generated_qa_dataset.csv";
        this.initialized = false;
    }

    async initialize() {
        if (!this.initialized) {
            await this.loadDocuments();
            this.initialized = true;
            console.log(`✅ Loaded ${this.documents.length} question-based documents into memory`);
        }
    }

    async loadCSV() {
        return new Promise((resolve, reject) => {
            const results = [];
            fs.createReadStream(this.csvFile)
                .pipe(csv())
                .on('data', (data) => results.push(data))
                .on('end', () => resolve(results))
                .on('error', reject);
        });
    }

    async loadDocuments() {
        try {
            console.log("📊 Loading Q&A CSV data...");
            const data = await this.loadCSV();
            
            this.documents = [];

            for (let index = 0; index < data.length; index++) {
                const row = data[index];

                // Handle AI-generated Q&A format (matching your Python version)
                if (row.question && row.answer) {
                    const question = String(row.question || '').trim();
                    const answer = String(row.answer || '').trim();
                    const source = row.source || 'unknown';

                    // Skip if question or answer is empty or too short
                    if (!question || question === 'nan' || question.length < 10) continue;
                    if (!answer || answer === 'nan' || answer.length < 10) continue;

                    const docId = `question_${index}`;

                    // Store ONLY the question as content for embedding (like Python version)
                    const questionContent = question;

                    const metadata = {
                        source: source,
                        question: question,
                        answer: answer,
                        qa_pair_id: index,
                        question_length: question.length,
                        answer_length: answer.length,
                        type: 'qa'
                    };

                    this.documents.push({
                        id: docId,
                        content: questionContent, // Only question for similarity matching
                        metadata: metadata,
                        tokens: this.tokenize(questionContent.toLowerCase())
                    });
                } else {
                    continue;
                }
            }

            console.log(`Created ${this.documents.length} question-based chunks`);

        } catch (error) {
            console.error("❌ Error loading documents:", error);
            throw error;
        }
    }

    tokenize(text) {
        // Enhanced tokenization for better question matching
        return text
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(token => token.length > 2)
            .map(token => token.toLowerCase());
    }

    calculateQuestionSimilarity(queryTokens, docTokens) {
        // Enhanced similarity calculation for questions
        const intersection = queryTokens.filter(token => docTokens.includes(token));
        const union = [...new Set([...queryTokens, ...docTokens])];
        
        // Jaccard similarity
        const jaccard = intersection.length / union.length;
        
        // Boost for common question words
        const questionWords = ['how', 'what', 'where', 'when', 'why', 'who', 'which'];
        let questionBoost = 0;
        for (const word of questionWords) {
            if (queryTokens.includes(word) && docTokens.includes(word)) {
                questionBoost += 0.1;
            }
        }
        
        return Math.min(jaccard + questionBoost, 1.0);
    }

    async similaritySearch(query, k = 5, threshold = 0.4) {
        try {
            await this.initialize();

            const queryTokens = this.tokenize(query.toLowerCase());
            const results = [];

            for (const doc of this.documents) {
                const similarity = this.calculateQuestionSimilarity(queryTokens, doc.tokens);
                
                if (similarity >= threshold) {
                    results.push({
                        document: doc.content, // This is just the question
                        metadata: doc.metadata, // This contains the full Q&A
                        similarity: similarity
                    });
                }
            }

            // Sort by similarity (descending) and return top k
            results.sort((a, b) => b.similarity - a.similarity);
            const topResults = results.slice(0, k);
            
            console.log(`📚 Found ${topResults.length} relevant questions with similarity >= ${threshold}`);
            return topResults;

        } catch (error) {
            console.error("❌ Error in similarity search:", error);
            return [];
        }
    }

    // Enhanced keyword search for questions
    async keywordSearch(query, k = 5) {
        try {
            await this.initialize();

            const queryLower = query.toLowerCase();
            const queryWords = queryLower.split(/\s+/).filter(word => word.length > 2);
            const results = [];

            for (const doc of this.documents) {
                const questionLower = doc.metadata.question.toLowerCase();
                const answerLower = doc.metadata.answer.toLowerCase();
                let score = 0;

                // Check for exact phrase matches in question
                if (questionLower.includes(queryLower)) {
                    score += 3;
                }

                // Check for exact phrase matches in answer
                if (answerLower.includes(queryLower)) {
                    score += 2;
                }

                // Check for individual keyword matches in question
                for (const word of queryWords) {
                    if (questionLower.includes(word)) {
                        score += 2;
                    }
                    if (answerLower.includes(word)) {
                        score += 1;
                    }
                }

                // Check source for relevant matches
                if (doc.metadata.source.toLowerCase().includes(queryLower)) {
                    score += 0.5;
                }

                if (score > 0) {
                    results.push({
                        document: doc.content, // Just the question
                        metadata: doc.metadata, // Full Q&A data
                        similarity: score / 10 // Normalize score
                    });
                }
            }

            // Sort by score (descending) and return top k
            results.sort((a, b) => b.similarity - a.similarity);
            const topResults = results.slice(0, k);
            
            console.log(`🔍 Keyword search found ${topResults.length} relevant results`);
            return topResults;

        } catch (error) {
            console.error("❌ Error in keyword search:", error);
            return [];
        }
    }

    // Combined search method (like Python retrieval)
    async questionSearch(query, k = 5, threshold = 0.4) {
        try {
            console.log(`🔍 Searching for questions similar to: "${query}"`);
            
            // Get results from both methods
            const similarityResults = await this.similaritySearch(query, k, threshold);
            const keywordResults = await this.keywordSearch(query, k);
            
            // Combine and deduplicate results
            const allResults = [...similarityResults, ...keywordResults];
            const uniqueResults = [];
            const seenIds = new Set();
            
            for (const result of allResults) {
                const id = result.metadata.qa_pair_id;
                if (!seenIds.has(id)) {
                    seenIds.add(id);
                    uniqueResults.push(result);
                }
            }
            
            // Sort by similarity score and get top k
            uniqueResults.sort((a, b) => b.similarity - a.similarity);
            const finalResults = uniqueResults.slice(0, k);
            
            console.log(`📊 Final results: ${finalResults.length} unique questions found`);
            
            // Log the questions found for debugging
            finalResults.forEach((result, index) => {
                console.log(`${index + 1}. ${result.metadata.question.substring(0, 60)}... (Score: ${result.similarity.toFixed(3)})`);
            });
            
            return finalResults;
            
        } catch (error) {
            console.error("❌ Error in question search:", error);
            return [];
        }
    }
}

module.exports = SimpleVectorStore;