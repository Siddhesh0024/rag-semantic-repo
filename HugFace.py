import streamlit as st
from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

st.set_page_config(page_title="RAG Analysis", layout="wide")

st.title("📄 Semantic Search & Q&A Engine (RAG)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # Extract text
    text = extract_text("temp.pdf")

    # Remove references
    if "References" in text:
        text = text.split("References")[0]

    # Extract Abstract
    abstract_match = re.search(r"Abstract[:\s]*(.*?)1\.Introduction", text, re.DOTALL)
    abstract = abstract_match.group(1) if abstract_match else ""

    body = text.replace(abstract, "")

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    body_chunks = text_splitter.split_text(body)
    chunks = [abstract] + body_chunks

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embedding_model
    )

    st.success("Document indexed successfully!")

    query = st.text_input("Ask a question about the document:")

    if query:

        docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Load LLM
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

        input_text = f"""
        Based on the following context, answer the question concisely.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("Answer:")
        st.write(answer)
