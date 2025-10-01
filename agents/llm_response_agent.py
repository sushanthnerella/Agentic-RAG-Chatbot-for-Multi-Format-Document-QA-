from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from core.models import MCPMessage, MCPPayload, HistoryMessage
from typing import List

# --- New Prompt for creating a standalone question ---
CONDENSE_QUESTION_PROMPT = """
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question that can be understood without the chat history.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""


# --- MODIFIED: Updated Prompt for generating the final answer ---
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant. Use the following context and chat history to answer the question.
Each piece of context is preceded by its source document. Pay close attention to this source information.

CHAT HISTORY:
{chat_history}

CONTEXT:
{context}

QUESTION:
{question}

Based on the chat history and the context provided (including the source of each piece of context), please provide a clear and concise answer. If the context does not contain the answer, state that the information is not available in the provided documents.
"""

def _format_chat_history(chat_history: List[HistoryMessage]) -> str:
    """Helper function to format chat history for the prompt."""
    if not chat_history:
        return "No history provided."
    return "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])


def condense_question(message: MCPMessage) -> str:
    """
    Takes chat history and a new question, and returns a standalone question.
    """
    query = message.payload.query
    chat_history = message.payload.chat_history

    if not chat_history:
        return query

    print("LLMResponseAgent: Condensing question with chat history...")
    
    prompt = PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT)
    llm = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0)

    condensing_chain = prompt | llm
    
    history_str = _format_chat_history(chat_history)
    
    standalone_question = condensing_chain.invoke({
        "chat_history": history_str,
        "question": query
    }).content

    print(f"LLMResponseAgent: Standalone question is: '{standalone_question}'")
    return standalone_question


def generate_response(message: MCPMessage) -> MCPMessage:
    """
    Handles a GENERATION_REQUEST by using an LLM to answer a query based on context and history.
    """
    query = message.payload.query
    context = message.payload.context
    chat_history = message.payload.chat_history
    
    if not query or context is None:
        raise ValueError("Query and context are required in the payload.")

    print(f"LLMResponseAgent: Received query '{query}' with {len(context)} context chunks.")

    context_str = "\n---\n".join(context)
    history_str = _format_chat_history(chat_history)
    
    prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.3) 

    rag_chain = prompt | llm

    print("LLMResponseAgent: Generating final answer from LLM...")
    answer = rag_chain.invoke({
        "context": context_str, 
        "chat_history": history_str,
        "question": query
    }).content
    print(f"LLMResponseAgent: Generated answer: '{answer}'")

    response_payload = MCPPayload(
        data=message.payload.data,
        query=query,
        context=context,
        chat_history=chat_history,
        answer=answer,
        sources=[chunk[:100] + "..." for chunk in context]
    )

    return MCPMessage(
        sender="LLMResponseAgent",
        receiver=message.sender,
        type="FINAL_RESPONSE",
        trace_id=message.trace_id,
        payload=response_payload
    )

