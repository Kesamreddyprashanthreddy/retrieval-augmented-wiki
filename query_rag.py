import os
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.callbacks import StreamingStdOutCallbackHandler

from embed_store import (
    load_vector_store, 
    create_embeddings,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL
)

LLM_MODEL = "llama3"
NUM_RESULTS = 4
TEMPERATURE = 0.7


def create_llm(model: str = LLM_MODEL, temperature: float = TEMPERATURE) -> Ollama:
    print(f"\n🤖 Initializing LLM: {model}")
    
    llm = Ollama(
        model=model,
        base_url="http://localhost:11434",
        temperature=temperature,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    print("✓ LLM initialized")
    return llm


def create_rag_prompt() -> PromptTemplate:
    template = """You are a helpful assistant answering questions about company documentation.

Use ONLY the following context to answer the question. If the answer is not in the context, 
say "I don't have enough information to answer this question based on the available documents."

Do not make up information. Be concise and accurate.
Do not reference document numbers or sources in your answer - just provide the answer directly.

Context:
{context}

Question: {question}

Answer:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def retrieve_relevant_chunks(
    vector_store: Chroma,
    query: str,
    k: int = NUM_RESULTS
) -> List[Tuple[Document, float]]:
    results = vector_store.similarity_search_with_score(query, k=k)
    return results


def format_context(documents: List[Document]) -> str:
    context_parts = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[Document {i+1}] (Source: {source})\n{doc.page_content}")
    
    return "\n\n---\n\n".join(context_parts)


def query_rag(
    question: str,
    vector_store: Chroma,
    llm: Ollama,
    num_results: int = NUM_RESULTS,
    show_sources: bool = True
) -> str:
    print("\n" + "=" * 50)
    print(f"❓ Question: {question}")
    print("=" * 50)
    
    # Step 1: Retrieve relevant chunks
    print(f"\n🔍 Retrieving top {num_results} relevant chunks...")
    results_with_scores = retrieve_relevant_chunks(vector_store, question, k=num_results)
    
    # Extract documents from results
    documents = [doc for doc, score in results_with_scores]
    
    if show_sources:
        print("\n📚 Retrieved Sources:")
        print("-" * 30)
        for i, (doc, score) in enumerate(results_with_scores):
            source = doc.metadata.get("source", "Unknown")
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"  [{i+1}] Score: {score:.4f}")
            print(f"      Source: {source}")
            print(f"      Preview: {preview}...")
            print()
    
    # Step 2: Format context
    context = format_context(documents)
    
    # Step 3: Create prompt
    prompt_template = create_rag_prompt()
    prompt = prompt_template.format(context=context, question=question)
    
    # Step 4: Generate answer
    print("\n💭 Generating answer...")
    print("-" * 30)
    
    answer = llm.invoke(prompt)
    
    print("\n" + "-" * 30)
    
    return answer


def create_rag_chain(vector_store: Chroma, llm: Ollama) -> RetrievalQA:
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": NUM_RESULTS}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": create_rag_prompt()}
    )
    
    return qa_chain


class RAGSystem:
    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        llm_model: str = LLM_MODEL,
        embedding_model: str = EMBEDDING_MODEL
    ):
        print("\n🚀 Initializing RAG System")
        print("=" * 50)
        
        self.embeddings = create_embeddings()
        
        self.vector_store = load_vector_store(
            self.embeddings,
            persist_directory,
            collection_name
        )
        
        self.llm = create_llm(llm_model)
        
        print("\n✅ RAG System ready!")
        print("=" * 50)
    
    def ask(self, question: str, show_sources: bool = True) -> str:
        return query_rag(
            question=question,
            vector_store=self.vector_store,
            llm=self.llm,
            show_sources=show_sources
        )
    
    def interactive_mode(self):
        print("\n" + "=" * 50)
        print("🎯 Interactive RAG Mode")
        print("=" * 50)
        print("Type your questions below. Type 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!")
                    break
                
                if not question:
                    print("Please enter a question.")
                    continue
                
                answer = self.ask(question)
                print(f"\n✅ Answer: {answer}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Local RAG System for Company Wiki",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_rag.py -q "What is the dress code?"
  python query_rag.py --interactive
  python query_rag.py -q "How do I request time off?" --no-sources
        """
    )
    
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Question to ask the RAG system"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start interactive Q&A mode"
    )
    
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Hide source documents in output"
    )
    
    args = parser.parse_args()
    
    try:
        rag = RAGSystem()
    except Exception as e:
        print(f"\n❌ Error initializing RAG system: {e}")
        print("\nMake sure:")
        print("1. Ollama is running (ollama serve)")
        print("2. Required models are pulled:")
        print("   - ollama pull nomic-embed-text")
        print("   - ollama pull llama3")
        print("3. Vector database exists (run embed_store.py first)")
        return
    
    if args.interactive:
        rag.interactive_mode()
    elif args.question:
        answer = rag.ask(args.question, show_sources=not args.no_sources)
        print(f"\n📝 Final Answer:\n{answer}")
    else:
        rag.interactive_mode()


if __name__ == "__main__":
    main()
