from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from agent import create_agent, chat
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import HumanMessage, AIMessage
import threading

app = FastAPI(title="Agent API",
             description="API for the LangChain Agent",
             version="1.0.0")

# CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent
agent_executor = create_agent()

# Initialize chat history and lock
chat_history: List[Tuple[str, str]] = []
chat_lock = threading.Lock()

# Ensure agent has memory attribute
if not hasattr(agent_executor, 'memory'):
    from langchain.memory import ConversationBufferMemory
    agent_executor.memory = ConversationBufferMemory(return_messages=True)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.get("/")
async def root():
    return {"message": "Agent API is running. Use /chat to interact with the agent."}

# In-memory session storage (in production, use a proper database)
session_storage = {}

def get_or_create_session(session_id: str) -> List[Tuple[str, str]]:
    if session_id not in session_storage:
        session_storage[session_id] = []
    return session_storage[session_id]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    try:
        # Generate a session ID if not provided
        if not chat_request.session_id:
            import uuid
            session_id = str(uuid.uuid4())
        else:
            session_id = chat_request.session_id
        
        with chat_lock:
            # Get or create session
            session = get_or_create_session(session_id)
            
            try:
                # Ensure agent has memory
                if not hasattr(agent_executor, 'memory'):
                    from langchain.memory import ConversationBufferMemory
                    agent_executor.memory = ConversationBufferMemory(return_messages=True)
                
                # Process the chat message
                response = chat(chat_request.message, agent_executor)
                
                # Get the updated history safely
                updated_history = []
                if hasattr(agent_executor, 'memory') and agent_executor.memory:
                    if hasattr(agent_executor.memory, 'chat_memory'):
                        updated_history = agent_executor.memory.chat_memory.messages
                
                # If no updated history, use the current session
                if not updated_history:
                    updated_history = session
                
                # Update session storage with the latest messages
                # Convert messages to a serializable format
                serialized_history = []
                for msg in updated_history:
                    if hasattr(msg, 'content'):
                        role = 'assistant' if hasattr(msg, 'type') and msg.type == 'ai' else 'user'
                        serialized_history.append((role, msg.content))
                
                if serialized_history:
                    session_storage[session_id] = serialized_history
                
                return {
                    "response": response,
                    "session_id": session_id
                }
                
            except Exception as e:
                # Log the full error for debugging
                import traceback
                error_details = traceback.format_exc()
                print(f"Error in chat processing: {error_details}")
                
                # Return a user-friendly error message
                return {
                    "response": "I'm sorry, I encountered an error processing your request. Please try again.",
                    "session_id": session_id
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)