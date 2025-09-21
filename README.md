# AI-Powered Ask Platform Bot

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](#)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#)

---

## 📖 Overview

**AI-Powered Ask Platform Bot** is a retrieval-augmented question-answering (QA) system that allows users to query documents and receive precise AI-generated answers. It leverages **vector databases (Chroma)**, **prompt templates**, and a **retrieval-augmented generation (RAG) pipeline** to ensure responses are contextual, accurate, and explainable.  

The project is designed to support integration with platforms like **Slack** and **Confluence**, making it a powerful tool for team knowledge management.

---

## ✨ Features

- 🔍 **Contextual QA with RAG** – Retrieves relevant information before generating responses.  
- 🧠 **Smart QA Tracker** – Avoids duplicate answers, tracks answered/unanswered questions.  
- 📚 **Multi-database support** – Integrates with multiple Chroma DB instances (e.g., Confluence, LangChain).  
- ⚡ **Configurable prompts** – Customize how the AI responds via prompt templates.  
- 🗄 **Persistent storage** – Uses SQLite (`page_tracking.db`) to track page/document usage.  
- 🤖 **Pluggable LLMs** – Works with Any LLM.  

---

## 🏗 Architecture / Components

| File / Module              | Description |
|-----------------------------|-------------|
| **`main.py`**              | Entry point for running the bot. |
| **`database.py`**          | Handles database interactions (storing/retrieving metadata). |
| **`qa_rag_pipeline.py`**   | Core retrieval-augmented generation pipeline. |
| **`smart_qa_tracker.py`**  | Tracks user queries, prevents duplicate answers, manages QA state. |
| **`prompts.py`**           | Contains system/user prompt templates for LLMs. |
| **`chroma_*_db/`**         | Vector database directories (Confluence, LangChain, QA storage). |
| **`page_tracking.db`**     | SQLite database for tracking page usage and QA history. |
| **`chroma.conf`**          | Configuration file for Chroma DB connections. |
| **`requirements.txt`**     | Python dependencies list. |

---

## ⚙️ Setup / Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sawy03/AI-powered-ask-platform-bot.git
   cd AI-powered-ask-platform-bot
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables**
   ```bash
   # Slack Configuration
   SLACK_BOT_TOKEN=your_token_here
   SLACK_SIGNING_SECRET=your_key_here
   APP_TOKEN=your_token_here
   USE_SOCKET_MODE=false
   
   
   # Confluence Configuration
   CONFLUENCE_BASE_URL=your_base_url_here
   CONFLUENCE_USERNAME=your_username_here
   CONFLUENCE_API_TOKEN=your_api_key_here

   # LLM Configuration
   EMBEDDINGS_MODEL=your_embeddings_model_name_here
   EMBEDDINGS_BASE_URL=your_embeddings_model_url_here //if you are hosting it somewhere
   LLM_MODEL=your_llm_model_name_here
   LLM_BASE_URL=your_llm_model_url_here //if you are hosting it somewhere
   
   CONFLUENCE_SPACE_KEYS=your_space_name_here
   PORT=your_port_here
   ```
4. **Run the bot**
   ```bash
   python main.py
   ```
