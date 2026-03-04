import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "wiki_documents"


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    print(f"\n📝 Chunking documents...")
    print(f"   Chunk size: {chunk_size} characters")
    print(f"   Chunk overlap: {chunk_overlap} characters")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"✓ Created {len(chunks)} chunks from {len(documents)} documents")
    print(f"   Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    
    return chunks


def create_embeddings() -> OllamaEmbeddings:
    print(f"\n🔢 Initializing embedding model: {EMBEDDING_MODEL}")
    print("   (Make sure Ollama is running with the model pulled)")
    
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:11434"
    )
    
    print("✓ Embedding model initialized")
    return embeddings


def create_vector_store(
    chunks: List[Document],
    embeddings: OllamaEmbeddings,
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME
) -> Chroma:
    print(f"\n💾 Creating vector store...")
    print(f"   Persist directory: {persist_directory}")
    print(f"   Collection name: {collection_name}")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    print(f"✓ Vector store created with {len(chunks)} vectors")
    print(f"✓ Data persisted to {persist_directory}")
    
    return vector_store


def load_vector_store(
    embeddings: OllamaEmbeddings,
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME
) -> Chroma:
    print(f"\n📂 Loading existing vector store from {persist_directory}")
    
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    collection = vector_store._collection
    count = collection.count()
    print(f"✓ Loaded vector store with {count} vectors")
    
    return vector_store


def build_vector_database(documents: List[Document]) -> Chroma:
    print("\n" + "=" * 50)
    print("🔨 Building Vector Database")
    print("=" * 50)
    
    chunks = chunk_documents(documents)
    embeddings = create_embeddings()
    vector_store = create_vector_store(chunks, embeddings)
    
    print("\n" + "=" * 50)
    print("✅ Vector database ready!")
    print("=" * 50)
    
    return vector_store


if __name__ == "__main__":
    from load_docs import load_all_documents
    
    DOCS_DIR = "./sample_docs"
    
    try:
        documents = load_all_documents(DOCS_DIR)
        
        if documents:
            vector_store = build_vector_database(documents)
            
            print("\n🔍 Testing similarity search...")
            query = "What is the dress code?"
            results = vector_store.similarity_search(query, k=2)
            
            print(f"\nQuery: '{query}'")
            print("\nTop results:")
            for i, doc in enumerate(results):
                print(f"\n[{i+1}] {doc.page_content[:200]}...")
        else:
            print("No documents found. Please add documents to the sample_docs directory.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. Ollama is running (ollama serve)")
        print("2. The embedding model is pulled (ollama pull nomic-embed-text)")
