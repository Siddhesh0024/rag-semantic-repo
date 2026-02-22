import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#print(text[:1000])
#from langchain.text_splitter import RecursiveCharacterTextSplitterdef 
def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

text = extract_text("/Users/siddheshhirve/Desktop/RAG LLM/CNS_research_paper copy.pdf")
if "References" in text:
    text = text.split("References")[0]
abstract = ""
if "Abstract" in text and "1.Introduction" in text:
    abstract = text.split("Abstract")[1].split("1.Introduction")[0]
body = text.replace(abstract, "")

print("Abstract extracted:")
print(abstract[:500])
print(text)
#text = extract_text("/Users/siddheshhirve/Desktop/RAG LLM/CNS_research_paper copy.pdf")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150
)
chunks = text_splitter.split_text(text)
print(f"Total chunks: {len(chunks)}")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_texts(
    texts=chunks,
    embedding=embedding_model
)

query = "What does this paper aim to achieve?"

docs = vector_store.similarity_search(query, k=3)

for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content[:500])
