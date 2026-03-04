import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader
)
from langchain_core.documents import Document


def load_markdown_files(directory: str) -> List[Document]:
    loader = DirectoryLoader(
        directory,
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} markdown files")
    return documents


def load_text_files(directory: str) -> List[Document]:
    loader = DirectoryLoader(
        directory,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} text files")
    return documents


def load_all_documents(directory: str) -> List[Document]:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    print(f"\n📂 Loading documents from: {directory}")
    print("-" * 50)
    
    all_documents = []
    
    try:
        md_docs = load_markdown_files(directory)
        all_documents.extend(md_docs)
    except Exception as e:
        print(f"⚠ Warning loading markdown files: {e}")
    
    try:
        txt_docs = load_text_files(directory)
        all_documents.extend(txt_docs)
    except Exception as e:
        print(f"⚠ Warning loading text files: {e}")
    
    print("-" * 50)
    print(f"📄 Total documents loaded: {len(all_documents)}")
    
    for doc in all_documents:
        if "source" not in doc.metadata:
            doc.metadata["source"] = "unknown"
    
    return all_documents


def preview_documents(documents: List[Document], num_chars: int = 200) -> None:
    print("\n📋 Document Preview:")
    print("=" * 50)
    
    for i, doc in enumerate(documents[:5]):
        source = doc.metadata.get("source", "unknown")
        content_preview = doc.page_content[:num_chars].replace("\n", " ")
        print(f"\n[{i+1}] Source: {source}")
        print(f"    Content: {content_preview}...")
    
    if len(documents) > 5:
        print(f"\n... and {len(documents) - 5} more documents")


if __name__ == "__main__":
    DOCS_DIR = "./sample_docs"
    
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    sample_md = """# Company Wiki

## Introduction
Welcome to our company wiki. This document contains important information.

## Policies
- Remote work is allowed 3 days per week
- All meetings should have an agenda
- Code reviews are required before merging

## Team Structure
We have the following teams:
- Engineering
- Product
- Design
- Operations
"""
    
    sample_txt = """FAQ Document

Q: What are the office hours?
A: Our office hours are 9 AM to 6 PM, Monday through Friday.

Q: How do I request time off?
A: Submit a request through the HR portal at least 2 weeks in advance.

Q: What is the dress code?
A: Business casual is the standard dress code.
"""
    
    with open(os.path.join(DOCS_DIR, "wiki.md"), "w", encoding="utf-8") as f:
        f.write(sample_md)
    
    with open(os.path.join(DOCS_DIR, "faq.txt"), "w", encoding="utf-8") as f:
        f.write(sample_txt)
    
    print("Created sample documents for testing")
    
    docs = load_all_documents(DOCS_DIR)
    preview_documents(docs)
