from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import re

# 1️⃣ Extract text
text = extract_text("CNS_research_paper copy.pdf")

# 2️⃣ Remove references section
if "References" in text:
    text = text.split("References")[0]

# 3️⃣ Extract Abstract properly using regex
abstract_match = re.search(r"Abstract[:\s]*(.*?)1\.Introduction", text, re.DOTALL)
abstract = ""

if abstract_match:
    abstract = abstract_match.group(1)

# Remove abstract from body
body = text.replace(abstract, "")

# 4️⃣ Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150
)

body_chunks = text_splitter.split_text(body)

# Combine abstract + body
chunks = [abstract] + body_chunks

print("Total chunks:", len(chunks))

# 5️⃣ Load embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 6️⃣ Create FAISS vector store
vector_store = FAISS.from_texts(
    texts=chunks,
    embedding=embedding_model
)

print("FAISS index created successfully!")

# 7️⃣ Ask Question
query = "What does this paper tries to solve?"

docs = vector_store.similarity_search(query, k=3)

print("\nRetrieved Context:\n")

for i, doc in enumerate(docs):
    print(f"\n--- Chunk {i+1} ---")
    print(doc.page_content[:500])

# 8️⃣ Combine context
context = "\n\n".join([doc.page_content for doc in docs])

# 9️⃣ Load local LLM
qa_pipeline = pipeline(
    "text-generation",
    model="google/flan-t5-small"
)

prompt = f"""
Answer the question using ONLY the context below.
If the answer is not found, say you don't know.

Context:
{context}

Question:
{query}
"""

result = qa_pipeline(prompt, max_length=300)

print("\nFinal Answer:\n")
print(result[0]['generated_text'])