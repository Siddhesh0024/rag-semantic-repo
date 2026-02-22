# 📄 Semantic Search & Q&A Engine (RAG)

## 🚀 Overview
This project is a Retrieval-Augmented Generation (RAG) based Q&A system built using Streamlit.

It allows users to:
- Upload a PDF
- Extract text
- Perform semantic search
- Ask questions about the document
- Generate answers using FLAN-T5

---

## 🛠 Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Transformers
- Sentence Transformers

---

## 📂 How It Works
1. PDF is uploaded
2. Text is extracted using pdfminer
3. Text is chunked
4. Embeddings are generated
5. FAISS performs similarity search
6. FLAN-T5 generates final answer

---

## ▶️ Run Locally

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py

---

## 📌 Future Improvements
- Deploy to Streamlit Cloud
- Improve model efficiency
- Add multi-document support