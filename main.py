from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from core.models import ChatRequest, ChatResponse, UploadResponse, MCPMessage, MCPPayload
from agents.retrieval_agent import retrieve_context
from agents.llm_response_agent import generate_response, condense_question # Import new function
from agents.ingestion_agent import process_documents
from dotenv import load_dotenv
import os

# --- App Initialization ---
load_dotenv()
app = FastAPI(
    title="Agentic RAG Chatbot API",
    description="An API for a multi-format document QA chatbot using an agentic architecture.",
    version="1.0.0"
)
UPLOAD_DIRECTORY = "./uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.post("/upload", response_model=UploadResponse)
async def upload_documents_endpoint(session_id: str, files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    saved_files = []
    session_path = os.path.join(UPLOAD_DIRECTORY, session_id)
    os.makedirs(session_path, exist_ok=True)
    for file in files:
        file_path = os.path.join(session_path, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        saved_files.append(file_path)
    try:
        process_documents(file_paths=saved_files, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process documents: {e}")
    return UploadResponse(
        message=f"Successfully uploaded and started processing {len(saved_files)} files.",
        filenames=[os.path.basename(p) for p in saved_files],
        session_id=session_id
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest):
    """
    Endpoint for conversational chat, managing the agentic flow.
    """
    try:
        # --- Step 1: Condense the question using history (LLM Agent) ---
        condense_request_payload = MCPPayload(
            data={"session_id": request.session_id},
            query=request.query,
            chat_history=request.chat_history
        )
        condense_request_msg = MCPMessage(
            sender="CoordinatorAgent",
            receiver="LLMResponseAgent",
            type="CONDENSE_REQUEST",
            payload=condense_request_payload
        )
        standalone_question = condense_question(condense_request_msg)

        # --- Step 2: Retrieve context using the standalone question (Retrieval Agent) ---
        retrieval_request_payload = MCPPayload(
            data={"session_id": request.session_id},
            query=standalone_question # Use the new, standalone question for retrieval
        )
        retrieval_request_msg = MCPMessage(
            sender="CoordinatorAgent",
            receiver="RetrievalAgent",
            type="RETRIEVAL_REQUEST",
            trace_id=condense_request_msg.trace_id,
            payload=retrieval_request_payload
        )
        context_response_msg = retrieve_context(retrieval_request_msg)

        # --- Step 3: Generate final answer using history, context, and original query (LLM Agent) ---
        if not context_response_msg.payload.context:
            return ChatResponse(
                answer="I could not find relevant information in the uploaded documents.",
                sources=[],
                session_id=request.session_id
            )

        # We pass the original query and history to the final generation step
        final_generation_payload = MCPPayload(
            data={"session_id": request.session_id},
            query=request.query, # Original question
            context=context_response_msg.payload.context,
            chat_history=request.chat_history # Full history
        )
        llm_request_msg = MCPMessage(
            sender="CoordinatorAgent",
            receiver="LLMResponseAgent",
            type="GENERATION_REQUEST",
            trace_id=context_response_msg.trace_id,
            payload=final_generation_payload
        )
        final_response_msg = generate_response(llm_request_msg)

        # --- Step 4: Return the final response ---
        return ChatResponse(
            answer=final_response_msg.payload.answer,
            sources=final_response_msg.payload.sources,
            session_id=request.session_id
        )

    except Exception as e:
        print(f"An error occurred during chat processing: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
