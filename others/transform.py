import pandas as pd
import json
import re
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time

# Initialize your AI model
model = OllamaLLM(model="llama3.2:1b")

# Create a specialized prompt for Q&A generation
qa_generation_prompt = """
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

Document Content:
{document_content}

Generated Q&A pairs:"""

prompt_template = ChatPromptTemplate.from_template(qa_generation_prompt)
chain = prompt_template | model

def generate_qa_from_document(document_content, filename, max_retries=3):
    """
    Generate Q&A pairs from a single document using AI
    """
    print(f"Generating Q&A for: {filename}")
    
    if len(document_content.strip()) < 50:
        print(f"  Skipping {filename} - content too short")
        return []
    
    # Truncate very long documents to avoid token limits
    if len(document_content) > 4000:
        document_content = document_content[:4000] + "..."
        print(f"  Truncated {filename} to 4000 characters")
    
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}")
            
            # Generate Q&A pairs using AI
            result = chain.invoke({"document_content": document_content})
            
            # Parse the generated Q&A pairs
            qa_pairs = parse_qa_response(result, filename)
            
            if qa_pairs:
                print(f"  ✅ Generated {len(qa_pairs)} Q&A pairs")
                return qa_pairs
            else:
                print(f"  ❌ No valid Q&A pairs found in attempt {attempt + 1}")
                
        except Exception as e:
            print(f"  ❌ Error in attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
    
    print(f"  ❌ Failed to generate Q&A pairs for {filename} after {max_retries} attempts")
    return []

def parse_qa_response(ai_response, filename):
    """
    Parse the AI response to extract Q&A pairs
    """
    qa_pairs = []
    
    # Split response into sections starting with "Q:"
    sections = re.split(r'\n(?=Q:)', ai_response)
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Look for Q: and A: pattern
        qa_match = re.match(r'Q:\s*(.*?)\s*A:\s*(.*)', section, re.DOTALL)
        
        if qa_match:
            question = qa_match.group(1).strip()
            answer = qa_match.group(2).strip()
            
            # Clean up the question and answer
            question = clean_text(question)
            answer = clean_text(answer)
            
            # Validate Q&A pair
            if len(question) > 10 and len(answer) > 20:
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'source': filename,
                    'generated_by': 'ai_model'
                })
    
    return qa_pairs

def clean_text(text):
    """
    Clean up generated text
    """
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove any remaining Q: or A: prefixes that might have been included
    text = re.sub(r'^[QA]:\s*', '', text)
    
    return text

def process_all_documents(input_csv_path, output_csv_path, batch_size=5):
    """
    Process all documents and generate Q&A pairs
    """
    print(f"Reading documents from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    all_qa_pairs = []
    processed_count = 0
    total_documents = len(df)
    
    print(f"Found {total_documents} documents to process")
    print("=" * 60)
    
    for index, row in df.iterrows():
        document_content = str(row.get('text', '')).strip()
        filename = row.get('filename', f'document_{index}')
        
        if not document_content or document_content == 'nan':
            print(f"Skipping empty document: {filename}")
            continue
        
        # Generate Q&A pairs for this document
        qa_pairs = generate_qa_from_document(document_content, filename)
        all_qa_pairs.extend(qa_pairs)
        
        processed_count += 1
        print(f"Progress: {processed_count}/{total_documents} documents processed")
        print(f"Total Q&A pairs so far: {len(all_qa_pairs)}")
        print("-" * 40)
        
        # Add a small delay to avoid overwhelming the model
        time.sleep(1)
        
        # Save progress periodically
        if processed_count % batch_size == 0:
            save_qa_pairs(all_qa_pairs, f"{output_csv_path}_backup_{processed_count}.csv")
            print(f"💾 Backup saved: {processed_count} documents processed")
    
    # Save final results
    final_df = save_qa_pairs(all_qa_pairs, output_csv_path)
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print(f"📊 Total documents processed: {processed_count}")
    print(f"📝 Total Q&A pairs generated: {len(all_qa_pairs)}")
    print(f"📁 Saved to: {output_csv_path}")
    print(f"📈 Average Q&A pairs per document: {len(all_qa_pairs)/processed_count:.1f}")
    
    return final_df

def save_qa_pairs(qa_pairs, output_path):
    """
    Save Q&A pairs to CSV
    """
    if not qa_pairs:
        print("No Q&A pairs to save")
        return None
    
    df = pd.DataFrame(qa_pairs)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(qa_pairs)} Q&A pairs to {output_path}")
    
    return df

def generate_sample_qa():
    """
    Generate a few sample Q&A pairs for testing
    """
    sample_content = """
    Docker is a containerization platform that allows developers to package applications and their dependencies into lightweight, portable containers. To install Docker on Ubuntu, first update your package index with 'sudo apt update', then install required packages with 'sudo apt install apt-transport-https ca-certificates curl software-properties-common'. Add Docker's GPG key and repository, then install Docker CE with 'sudo apt install docker-ce'. After installation, add your user to the docker group with 'sudo usermod -aG docker $USER' to run Docker commands without sudo.
    """
    
    print("Testing Q&A generation with sample content...")
    qa_pairs = generate_qa_from_document(sample_content, "docker_install_guide.md")
    
    print("\nGenerated Q&A pairs:")
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\n{i}. Q: {qa['question']}")
        print(f"   A: {qa['answer'][:100]}...")

# Main execution
if __name__ == "__main__":
    print("🤖 AI-Powered Q&A Generator")
    print("=" * 60)
    
    # Configuration
    input_file = "all_documents2.csv"  # Your current document CSV
    output_file = "ai_generated_qa_dataset2.csv"  # New Q&A CSV
    
    # Test with sample first (optional)
    test_sample = input("\nTest with sample content first? (y/n): ").lower().strip()
    if test_sample == 'y':
        generate_sample_qa()
        continue_prompt = input("\nContinue with full processing? (y/n): ").lower().strip()
        if continue_prompt != 'y':
            print("Exiting...")
            exit()
    
    print(f"\nStarting full document processing...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    try:
        final_df = process_all_documents(input_file, output_file)
        
        if final_df is not None and len(final_df) > 0:
            print("\n📋 Sample of generated Q&A pairs:")
            print(final_df[['question', 'source']].head(5))
            
            print(f"\n📊 Q&A pairs by source:")
            source_counts = final_df['source'].value_counts()
            print(source_counts.head(10))
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {input_file}")
        print("Make sure your CSV file exists and the path is correct.")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
    
    print("\n✅ Process completed!")