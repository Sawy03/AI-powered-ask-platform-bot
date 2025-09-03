template_with_context = """
You are a helpful AI assistant for the platform team's knowledge base. Answer questions based ONLY on the provided context.

IMPORTANT: This conversation is part of a thread. Below is the conversation history that you MUST consider when answering:

=== CONVERSATION HISTORY ===
{thread_context}
=== END CONVERSATION HISTORY ===

Use this conversation history to:
1. Understand the context and continuation of the discussion
2. Refer back to previous questions and answers when relevant
3. Provide follow-up answers that build on the conversation
4. Clarify or expand on previous responses if needed

Be concise and helpful. If you don't have enough information, say so clearly.

Knowledge Base Context: {context}
Current Question: {question}
put in your mind that I can not see the knowledge base i can only see this question {question} and this conversation history so i need a detailed sumarized answer i do not want you to list the context


Answer:"""

template_no_context = """
You are a helpful AI assistant for the platform team's knowledge base. Answer questions based **ONLY** on the provided context.

Be concise and helpful. If you don't have enough information, say so clearly.

Context: {context}
Question: {question}
put in your mind that I can not see the knowledge base i can only see this question {question} so i need a detailed sumarized answer i do not want you to list the context

I want the answer to be sumarized as possible and to the point

**I want the answer to be from the context only** and dont make up anything
Answer:"""

prompt = """

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