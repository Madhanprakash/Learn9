
import os
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(page_title="Multi-PDF RAG", page_icon="📚")
st.title("📚 Multi-PDF RAG Assistant")
st.write("Upload multiple PDFs and ask questions across all of them.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing. Add it to your .env file.")
    st.stop()


uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)


if "store" not in st.session_state:
    st.session_state.store = None

if "file_signature" not in st.session_state:
    st.session_state.file_signature = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


if uploaded_files:
    # Detect changes to the uploaded PDFs
    file_hashes = []
    for uploaded_file in uploaded_files:
        file_hashes.append(
            (
                uploaded_file.name,
                hashlib.sha256(uploaded_file.getvalue()).hexdigest(),
            )
        )

    signature = tuple(sorted(file_hashes))

    if signature != st.session_state.file_signature:
        with st.spinner("Reading PDFs and building the vector index..."):
            all_chunks = []

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=120,
            )

            for uploaded_file in uploaded_files:
                # PyPDFLoader needs a filesystem path
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                try:
                    pages = PyPDFLoader(temp_path).load()

                    for page in pages:
                        # Replace temporary path with the real filename
                        page.metadata["source"] = uploaded_file.name

                        # PyPDFLoader page metadata is zero-based
                        page.metadata["page"] = int(
                            page.metadata.get("page", 0)
                        )

                    all_chunks.extend(
                        splitter.split_documents(pages)
                    )
                finally:
                    os.unlink(temp_path)

            if not all_chunks:
                st.error("No readable text was found in the uploaded PDFs.")
                st.stop()

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )

            st.session_state.store = FAISS.from_documents(
            all_chunks, embeddings
            )
            st.session_state.chunk_count = len(all_chunks)
            st.session_state.file_signature = signature

        st.success(
            f"Indexed {st.session_state.chunk_count} chunks "
            f"from {len(uploaded_files)} PDF(s)."
        )

    else:
        st.success(
            f"Index ready: {st.session_state.chunk_count} chunks."
        )

else:
    # Clear the old index if the user removes all uploads
    st.session_state.store = None
    st.session_state.file_signature = None
    st.session_state.chunk_count = 0
    st.info("Upload one or more PDFs to get started.")


if st.session_state.store is not None:
    question = st.text_input(
        "Ask a question about your documents"
    )

    if st.button("Get Answer", type="primary") and question.strip():
        retriever = st.session_state.store.as_retriever(
            search_kwargs={"k": 6}
        )
        docs = retriever.invoke(question)

        if not docs:
            st.warning("No relevant passages were retrieved.")
            st.stop()

        context_parts = []

        for doc in docs:
            source = Path(
                doc.metadata.get("source", "Unknown")
            ).name
            page = int(doc.metadata.get("page", 0)) + 1

            context_parts.append(
                f"Source: {source}, page {page}\n"
                f"{doc.page_content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

        messages = [
            (
                "system",
                """
You answer questions using only the supplied PDF excerpts.

Rules:
- Do not invent facts that are not supported by the excerpts.
- If the evidence is insufficient, say so clearly.
- If sources disagree, explicitly identify the disagreement.
- For each conflicting claim, name the document and page.
- Do not silently choose one conflicting source as correct.
- Distinguish direct evidence from inference.
- Cite supporting claims inline using [filename, page N].
""",
            ),
            (
                "human",
                f"PDF excerpts:\n{context}\n\nQuestion: {question}",
            ),
        ]

        with st.spinner("Generating your answer..."):
            response = llm.invoke(messages)

        st.subheader("Answer")
        st.markdown(response.content)

        st.subheader("Retrieved Sources")

        unique_sources = {}
        for doc in docs:
            source = Path(
                doc.metadata.get("source", "Unknown")
            ).name
            page = int(doc.metadata.get("page", 0)) + 1

            unique_sources[(source, page)] = doc.page_content

        for (source, page), text in unique_sources.items():
            with st.expander(f"{source} — Page {page}"):
                st.write(text)
