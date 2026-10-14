
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Question
question = "What is a token?"

# Step 1: Load document
file_path = BASE_DIR / "llm_document.txt"

if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")

loader = TextLoader(str(file_path), encoding="utf-8")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step 3: Create embeddings
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Store vectors in FAISS
vector_store = FAISS.from_documents(chunks, embeddings)
print("FAISS vector store created")

# Step 5: Retrieve relevant chunks
results = vector_store.similarity_search(question, k=3)
context = "\n\n".join(r.page_content for r in results)

print("\nQUESTION:", question)

# Step 6: Generate answer using OpenAI, if configured
api_key = os.environ.get("OPENAI_API_KEY")

if api_key:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the supplied context. "
                    "If the context does not contain the answer, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )

    print("\nANSWER:")
    print(response.choices[0].message.content)

else:
    print("\nNo OPENAI_API_KEY configured.")
    print("Retrieved context:\n")
    print(context)
