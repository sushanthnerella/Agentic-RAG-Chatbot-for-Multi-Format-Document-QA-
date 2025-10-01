from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
import chromadb
from core.models import MCPMessage, MCPPayload
import os
from typing import List

# --- Configuration ---
CHROMA_PERSIST_DIRECTORY = "chroma_db"

# --- New Prompt for Multi-Query Generation ---
MULTI_QUERY_PROMPT_TEMPLATE = """
You are an expert at generating search queries. Given a user's question, generate three additional versions of the question that are semantically similar but use different phrasing.
The goal is to improve the recall of a vector database search.
Return a comma-separated list of the queries. Do not include the original question.

Original Question: {question}

Generated Queries (comma-separated):
"""

# --- Re-ranking Prompt  ---
RERANK_PROMPT_TEMPLATE = """
You are an expert at re-ranking documents based on their relevance to a question.
Given a question and a list of documents (with their index), your task is to identify the indices of the FIVE most relevant documents.
Return a comma-separated list of the top 5 indices. Do not include any other text.

Question: {question}

Documents:
{documents}

Top 5 Indices (comma-separated):
"""

def _generate_search_queries(query: str) -> List[str]:
    """Uses a lightweight LLM to generate alternative search queries."""
    print(f"MultiQuery: Generating alternative queries for '{query}'...")
    prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT_TEMPLATE)
    llm = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0.7)
    output_parser = CommaSeparatedListOutputParser()
    chain = prompt | llm | output_parser

    try:
        generated_queries = chain.invoke({"question": query})
        print(f"MultiQuery: Generated {generated_queries}")
        # Return the original query plus the generated ones
        return [query] + generated_queries
    except Exception as e:
        print(f"MultiQuery: Error during query generation: {e}. Using original query only.")
        return [query]


def _rerank_documents(query: str, documents: List[str]) -> List[str]:
    """Uses a lightweight LLM to re-rank documents for relevance to a query."""
    if not documents:
        return []

    print(f"Reranker: Starting re-ranking for {len(documents)} documents.")
    
    formatted_docs = "\n\n".join([f"Index: {i}\nContent: {doc}" for i, doc in enumerate(documents)])
    prompt = PromptTemplate.from_template(RERANK_PROMPT_TEMPLATE)
    llm = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0)
    output_parser = CommaSeparatedListOutputParser()
    chain = prompt | llm | output_parser

    try:
        relevant_indices_str = chain.invoke({"question": query, "documents": formatted_docs})
        relevant_indices = [int(i.strip()) for i in relevant_indices_str]
        reranked_docs = [documents[i] for i in relevant_indices if i < len(documents)]
        print(f"Reranker: Successfully re-ranked. Top indices: {relevant_indices}")
        return reranked_docs
    except Exception as e:
        print(f"Reranker: Error during re-ranking: {e}. Returning original top 5 documents.")
        return documents[:5]


def retrieve_context(message: MCPMessage) -> MCPMessage:
    """
    Handles a RETRIEVAL_REQUEST by generating multiple queries, searching ChromaDB, re-ranking, and returning the best context.
    """
    session_id = message.payload.data.get("session_id")
    query = message.payload.query

    if not session_id or not query:
        raise ValueError("session_id and query are required in the payload.")

    print(f"RetrievalAgent: Received query '{query}' for session '{session_id}'")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)

    try:
        collection = client.get_collection(name=session_id)
    except ValueError:
        return MCPMessage(
            sender="RetrievalAgent", receiver=message.sender, type="CONTEXT_RESPONSE",
            trace_id=message.trace_id, payload=MCPPayload(data={}, context=[], query=query)
        )

    # Step 1: Generate multiple queries
    search_queries = _generate_search_queries(query)
    
    # Step 2: Retrieve documents for all queries
    all_results_docs = []
    all_results_metadatas = []
    for q in search_queries:
        results = collection.query(
            query_texts=[q],
            n_results=5, # Retrieve 5 docs per query
            include=["documents", "metadatas"]
        )
        all_results_docs.extend(results.get('documents', [[]])[0])
        all_results_metadatas.extend(results.get('metadatas', [[]])[0])

    # Remove duplicates while preserving order
    unique_docs = {}
    for i, doc in enumerate(all_results_docs):
        if doc not in unique_docs:
            unique_docs[doc] = all_results_metadatas[i]
    
    initial_chunks_with_metadata = []
    for doc, metadata in unique_docs.items():
        source = os.path.basename(metadata.get('source', 'Unknown Source'))
        formatted_chunk = f"Source: {source}\n\nContent: {doc}"
        initial_chunks_with_metadata.append(formatted_chunk)

    print(f"RetrievalAgent: Collected {len(initial_chunks_with_metadata)} unique chunks from multi-query search.")

    # Step 3: Re-rank the combined, unique documents
    reranked_chunks = _rerank_documents(query=query, documents=initial_chunks_with_metadata)
    
    print(f"RetrievalAgent: Final context count after re-ranking: {len(reranked_chunks)}.")

    response_payload = MCPPayload(
        data=message.payload.data,
        query=query,
        context=reranked_chunks
    )

    return MCPMessage(
        sender="RetrievalAgent", receiver=message.sender, type="CONTEXT_RESPONSE",
        trace_id=message.trace_id, payload=response_payload
    )

