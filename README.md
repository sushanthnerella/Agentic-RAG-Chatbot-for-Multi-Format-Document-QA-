# Agentic-RAG-Chatbot-for-Multi-Format-Document-QA-
# 📄 Agentic RAG Chatbot

A **multi-agent Retrieval-Augmented Generation (RAG) chatbot** capable of answering questions from multiple uploaded documents (PDF, DOCX, TXT, CSV, PPTX).

The system is built on a **FastAPI backend** (with 3 core agents) and a **Streamlit frontend** for user interaction. It uses **Google Gemini LLMs**, **ChromaDB vector store**, and a **Model Context Protocol (MCP)** for clean agent communication.

---

## 🚀 Features

* 📂 Upload and process documents in multiple formats.
* 🔎 Multi-query retrieval with re-ranking for **highly relevant answers**.
* 🧠 Agent-based architecture: Ingestion, Retrieval, and LLM Response.
* 💬 Multi-turn, conversational interface (Streamlit).
* 🗄️ Persistent ChromaDB for embeddings.

---

## 🏗️ Architecture

```
Frontend (Streamlit UI)  <──>  FastAPI Backend
                                   │
         ┌───────────────────────────────────────────┐
         │ CoordinatorAgent (main.py)                │
         │    ├── IngestionAgent: Parse & embed docs │
         │    ├── RetrievalAgent: Search & rerank    │
         │    └── LLMResponseAgent: Generate answers │
         └───────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
/agentic-rag-chatbot/
├── agentic_rag_backend/
│   ├── agents/
│   │   ├── ingestion_agent.py
│   │   ├── retrieval_agent.py
│   │   └── llm_response_agent.py
│   ├── core/
│   │   └── models.py
│   ├── chroma_db/
│   ├── uploaded_files/
│   ├── main.py
│   └── requirements.txt
└── streamlit_ui/
    └── app.py
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/agentic-rag-chatbot.git
cd agentic-rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r agentic_rag_backend/requirements.txt
```

### 4. Configure API Key

Create a `.env` file inside `agentic_rag_backend/` and add:

```
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

---

## ▶️ Running the Application

### Start the Backend (FastAPI)

```bash
cd agentic_rag_backend
uvicorn main:app --reload
```

### Start the Frontend (Streamlit)

```bash
cd streamlit_ui
streamlit run app.py
```

App will be available at **[http://localhost:8501](http://localhost:8501)**.

---

## 📘 Usage

1. Upload documents from the **sidebar**.
2. Ask questions in the chatbox.
3. Get **answers with cited sources** from your uploaded files.

---

## 📊 Tech Stack

* **FastAPI** – Backend API
* **Streamlit** – Frontend UI
* **LangChain + Google Gemini** – Embeddings & LLM responses
* **ChromaDB** – Vector database
* **Python-dotenv** – Environment configuration

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## 📜 License

This project is licensed under the MIT License.

---


# Agentic-RAG-Chatbot-for-Multi-Format-Document-QA-
