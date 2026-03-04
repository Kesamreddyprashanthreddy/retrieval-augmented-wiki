# Local RAG System for Company Wiki

A complete, locally-running Retrieval Augmented Generation (RAG) system for answering questions about your company's wiki documents.

## 🎯 Overview

This system allows you to:

- Load documents from Notion exports (markdown/text files)
- Create searchable embeddings using local models
- Query your documents using natural language
- Get AI-generated answers based on your actual content

**Everything runs locally - no paid APIs, no data leaving your machine.**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Notion  │───▶│  Chunk   │───▶│ Embed    │───▶│ ChromaDB │  │
│  │  Docs    │    │  Text    │    │ (Ollama) │    │ (Vector) │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                       │         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐        │         │
│  │  Answer  │◀───│  LLM     │◀───│ Retrieve │◀───────┘         │
│  │          │    │ (Ollama) │    │ Context  │                   │
│  └──────────┘    └──────────┘    └──────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
rag_project/
├── load_docs.py      # Document loading from Notion exports
├── embed_store.py    # Chunking, embedding, and vector storage
├── query_rag.py      # Query pipeline and answer generation
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and Start Ollama

Download Ollama from [ollama.ai](https://ollama.ai) and install it.

Start the Ollama server:

```bash
ollama serve
```

### 3. Pull Required Models

```bash
# Embedding model (for converting text to vectors)
ollama pull nomic-embed-text

# LLM model (for generating answers)
ollama pull llama3
```

### 4. Prepare Your Documents

Place your Notion export files (`.md` or `.txt`) in a directory, e.g., `./sample_docs/`

### 5. Build the Vector Database

```bash
# This will load, chunk, embed, and store your documents
python load_docs.py      # Creates sample docs for testing
python embed_store.py    # Builds the vector database
```

### 6. Query Your Documents

```bash
# Ask a single question
python query_rag.py -q "What is the dress code?"

# Start interactive mode
python query_rag.py --interactive

# Hide source documents in output
python query_rag.py -q "How do I request time off?" --no-sources
```

## 📚 Key Concepts Explained

### Why Chunking is Necessary

**Problem**: LLMs have limited context windows (e.g., 4K-128K tokens). A company wiki might have millions of characters.

**Solution**: Split documents into smaller, manageable chunks.

**Benefits**:

1. **Fits in context**: Chunks are small enough to include multiple in a prompt
2. **Precision**: Retrieve only relevant sections, not entire documents
3. **Better embeddings**: Focused text produces more meaningful vectors
4. **Efficiency**: Process and search smaller units faster

**How it works**:

```
Original Document (10,000 chars)
         │
         ▼
┌────────────────────┐
│ RecursiveCharacter │
│   TextSplitter     │
└────────────────────┘
         │
         ▼
┌───────┬───────┬───────┬───────┐
│Chunk 1│Chunk 2│Chunk 3│Chunk 4│  (1000 chars each)
│       │       │       │       │  (200 char overlap)
└───────┴───────┴───────┴───────┘
```

### Why Embeddings are Used

**Problem**: Computers can't understand text meaning directly.

**Solution**: Convert text to numerical vectors that capture semantic meaning.

**How embeddings work**:

```
"What is the dress code?"  ──▶  [0.12, -0.45, 0.78, ..., 0.33]  (768 dimensions)
"Office attire policy"     ──▶  [0.11, -0.43, 0.76, ..., 0.31]  (similar vector!)
"Best pizza recipes"       ──▶  [-0.89, 0.22, -0.15, ..., 0.67] (different vector)
```

**Key insight**: Similar meanings = similar vectors, regardless of exact words used.

### How Vector Similarity Search Works

**Process**:

1. **Indexing**: Convert all document chunks to vectors, store in ChromaDB
2. **Query**: Convert user's question to a vector
3. **Search**: Find stored vectors closest to the query vector
4. **Return**: Get the original text of the closest vectors

**Distance metrics**:

- **Cosine similarity**: Measures angle between vectors (most common)
- **L2 (Euclidean)**: Measures straight-line distance

```
Query Vector: Q ────────────────────▶ [0.5, 0.3, 0.8]
                                           │
                                           │ Find nearest
                                           ▼
Stored Vectors:          ┌─────────────────────────────────┐
  Doc1: [0.4, 0.3, 0.7]  │  Cosine Similarity: 0.98 ✓ TOP │
  Doc2: [0.1, 0.9, 0.2]  │  Cosine Similarity: 0.45       │
  Doc3: [0.5, 0.2, 0.9]  │  Cosine Similarity: 0.95 ✓     │
                         └─────────────────────────────────┘
```

### How Retrieved Context Improves LLM Answers

**Without RAG**:

```
User: "What's our remote work policy?"
LLM: "Typically companies allow 2-3 days..." (generic, possibly wrong)
```

**With RAG**:

```
User: "What's our remote work policy?"
     │
     ▼
[Retrieve from vector DB]
     │
     ▼
Context: "Remote work is allowed 3 days per week..."
     │
     ▼
LLM + Context: "According to your company policy, remote work
               is allowed 3 days per week." (accurate!)
```

**Benefits of RAG**:

1. **Accuracy**: Answers grounded in actual documents
2. **No hallucination**: LLM cites real content, not made-up facts
3. **Up-to-date**: Works with documents newer than LLM training
4. **Domain-specific**: Handles your unique company terminology
5. **Verifiable**: You can check the source documents

## ⚙️ Configuration

Edit the constants in `embed_store.py`:

```python
CHUNK_SIZE = 1000          # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks
EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_PERSIST_DIR = "./chroma_db"
```

Edit the constants in `query_rag.py`:

```python
LLM_MODEL = "llama3"       # Or "mistral", "codellama", etc.
NUM_RESULTS = 4            # Chunks to retrieve per query
TEMPERATURE = 0.7          # 0 = deterministic, 1 = creative
```

## 🔧 Troubleshooting

### "Connection refused" error

- Make sure Ollama is running: `ollama serve`

### "Model not found" error

- Pull the required models:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3
  ```

### "No documents found" error

- Check your documents directory path
- Ensure files have `.md` or `.txt` extensions

### Slow performance

- First query is slow due to model loading
- Subsequent queries are faster
- Consider using a smaller LLM like `mistral` or `phi`

## 📝 License

MIT License - Use freely for any purpose.
