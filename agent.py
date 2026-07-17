import requests
import uuid

from deepagents import StateBackend
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings



DOCS_BASE = "https://docs.langchain.com"

# Curated LangChain OSS pages for this tutorial. Expand this list or parse
# URLs from https://docs.langchain.com/llms.txt to index more of the site.
DOC_PATHS = [
    "oss/python/langchain/agents",
    "oss/python/deepagents/rag",
    "oss/python/langchain/tools",
    "oss/python/langchain/models",
    "oss/python/langchain/retrieval",
    "oss/python/langchain/knowledge-base",
    "oss/python/langchain/middleware",
    "oss/python/deepagents/overview",
    "oss/python/deepagents/subagents",
    "oss/python/deepagents/streaming",
    "oss/python/deepagents/frontend/subagent-streaming",
    "oss/python/deepagents/backends",
    "oss/python/langgraph/overview",
    "oss/python/langgraph/quickstart",
]


#  Load LangChain documentation pages as Documents
def load_langchain_docs(doc_paths: list[str] | None = None) -> list[Document]:
    """Fetch LangChain documentation pages as Documents."""
    paths = doc_paths or DOC_PATHS
    docs: list[Document] = []
    for path in paths:
        url = f"{DOCS_BASE}/{path}.md"
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
        except requests.RequestException:
            continue
        source = f"{DOCS_BASE}/{path}"
        docs.append(
            Document(page_content=response.text, metadata={"source": source})
        )
    return docs


docs = load_langchain_docs()
print(f"Loaded {len(docs)} documentation pages.")

#  Split documents into chunks for embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs_chunks = text_splitter.split_documents(docs)
print(f"Split into {len(docs_chunks)} chunks.")


#  Create embeddings and store in a vector store
#  1.Google Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

#  Store/index embeddings and chunks in an in-memory vector store
# vector_store = InMemoryVectorStore.from_documents(docs_chunks, embeddings)
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=docs_chunks)

print(f"Indexed {len(docs_chunks)} chunks.")

#  Build the agent
#  Search tool for the agent to use

backend = StateBackend()

@tool(parse_docstring=True)
def search_documentation(query: str) -> str:
    """Search LangChain documentation and save matching chunks to the agent filesystem.

    Args:
        query: Natural language search query.

    Returns:
        File paths where retrieved chunks were saved under /retrieved/.
    """
    retrieved_docs = vector_store.similarity_search(query, k=4)
    batch_id = uuid.uuid4().hex[:8]
    uploads: list[tuple[str, bytes]] = []
    saved_paths: list[str] = []

    for index, doc in enumerate(retrieved_docs, start=1):
        path = f"/retrieved/{batch_id}/chunk_{index}.md"
        content = (
            f"# Source: {doc.metadata.get('source', 'unknown')}\n\n"
            f"{doc.page_content}"
        )
        uploads.append((path, content.encode("utf-8")))
        saved_paths.append(path)

    backend.upload_files(uploads)
    return (
        f"Saved {len(saved_paths)} documentation chunks:\n"
        + "\n".join(saved_paths)
    )

