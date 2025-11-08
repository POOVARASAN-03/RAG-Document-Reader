# 📄 ChatPDF RAG App  

An **interactive RAG-based LLM-powered assistant** for querying across **multiple documents (PDF, DOCX, PPTX)** using Google's Gemini model and FAISS vector store.  
Built with 🧠 **LangChain**, 🌐 **Streamlit**, and 🔍 **Google Generative AI**.

---

## 🚀 Live Demo  

👉 **Try it here:** [https://rag-multi-doc-reader.streamlit.app/](https://rag-multi-doc-reader.streamlit.app/)

---

## ✨ Features  

- 📚 **Multi-format Support** – Upload and query from **PDF**, **DOCX**, and **PPTX** files.  
- ⚡ **Fast & Accurate Responses** – Powered by **Google Gemini (gemini-2.0-flash-exp)** for real-time answers.  
- 🧩 **Smart Document Chunking** – Efficient text splitting and embedding via **LangChain**.  
- 🔍 **Semantic Search** – Uses **FAISS Vector Store** for contextually relevant retrieval.  
- 💬 **Interactive Q&A** – Chat with your uploaded documents directly through the Streamlit interface.  

---

## 🧠 Tech Stack  

- **Frontend:** Streamlit  
- **LLM:** Google Gemini  
- **Framework:** LangChain  
- **Vector Database:** FAISS  
- **Embeddings:** Google Generative AI Embeddings  

---

## 📦 Installation  

Make sure you have **Python 3.8+** installed, then run:  

```bash
pip install -r requirements.txt

## ▶️ Usage

Run the Streamlit app locally:

streamlit run app.py


Then open http://localhost:8501
 in your browser.

## ⚙️ Environment Setup

Create a .env file and add your Google API Key:

GOOGLE_API_KEY=your_api_key_here

## 📁 Supported File Types
Format	Description	Library Used
📘 PDF	Portable Document Format	PyPDF2
📝 DOCX	Microsoft Word Document	python-docx
📊 PPTX	PowerPoint Presentation	python-pptx
📚 Example Use Cases


## 🤝 Contributing

Pull requests and feature suggestions are welcome!
Fork the repo, make your changes, and submit a PR 🚀