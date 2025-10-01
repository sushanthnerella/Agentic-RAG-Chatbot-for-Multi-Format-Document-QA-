from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid

# --- New model for a single message in the chat history ---
class HistoryMessage(BaseModel):
    role: str # Will be 'user' or 'assistant'
    content: str

# --- API Request & Response Models ---

class ChatRequest(BaseModel):
    """Defines the structure for a user's chat message."""
    query: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # Add the chat history to the request
    chat_history: List[HistoryMessage] = []

class ChatResponse(BaseModel):
    """Defines the structure for the chatbot's response."""
    answer: str
    sources: List[str]
    session_id: str

class UploadResponse(BaseModel):
    """Defines the response after a successful file upload."""
    message: str
    filenames: List[str]
    session_id: str


# --- MCP (Model Context Protocol) Models ---

class MCPPayload(BaseModel):
    """A flexible payload for our internal agent communication."""
    data: Dict[str, Any]
    query: Optional[str] = None
    context: Optional[List[str]] = None
    answer: Optional[str] = None
    sources: Optional[List[str]] = None
    # Add chat history to our internal messages as well
    chat_history: Optional[List[HistoryMessage]] = None


class MCPMessage(BaseModel):
    """
    Defines the standard message structure for communication between agents.
    This is our implementation of the Model Context Protocol.
    """
    sender: str
    receiver: str
    type: str # e.g., "INGEST_REQUEST", "RETRIEVAL_REQUEST", "LLM_RESPONSE"
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    payload: MCPPayload
