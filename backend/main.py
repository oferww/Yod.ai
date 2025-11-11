from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
import time
from datetime import datetime
import logging
import logging.handlers
from pathlib import Path
import cohere
from dotenv import load_dotenv
from chatbot import OferGPT
from utils.chat_history_logger import get_chat_history_logger
from utils.semantic_rag import SemanticRAG
from utils.metrics_tracker import get_metrics_tracker

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "app.log"

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=5*1024*1024, backupCount=5
)
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
console_format = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
file_format = logging.Formatter(log_format)
console_handler.setFormatter(console_format)
file_handler.setFormatter(file_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load environment variables
load_dotenv()

# Configure chat history logging directory
# Use /app/chat_logs in Docker, or chat_logs in local development
chat_logs_dir = os.getenv("CHAT_LOGS_DIR", "/app/chat_logs")
if not os.path.exists("/app"):
    # Local development - use relative path
    chat_logs_dir = "chat_logs"

# Load products data
# Try multiple paths to find products.json
possible_paths = [
    os.path.join(os.path.dirname(__file__), "..", "products.json"),  # backend/../products.json
    os.path.join(os.path.dirname(__file__), "products.json"),  # backend/products.json
    "/app/products.json",  # Docker path
    "products.json",  # Current directory
    "../products.json"  # Docker path
]

PRODUCTS = []
PRODUCTS_FILE = None

for path in possible_paths:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                PRODUCTS = json.load(f)
            PRODUCTS_FILE = path
            print(f"Loaded {len(PRODUCTS)} products from {path}", flush=True)
            break
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in products file at {path}: {e}", flush=True)
        except Exception as e:
            print(f"Error loading products from {path}: {e}", flush=True)

if not PRODUCTS:
    print(f"Could not load products from any of these paths: {possible_paths}", flush=True)
    PRODUCTS = []

# Global SemanticRAG instance (initialized at startup)
semantic_rag = None

app = FastAPI(title="Laptop Recommendation Chatbot")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize SemanticRAG at application startup.
    
    This ensures the embedding model is downloaded and cached before
    the first user query, eliminating the first-query latency.
    """
    global semantic_rag
    
    print("[STARTUP] Initializing SemanticRAG system...", flush=True)
    if PRODUCTS:
        try:
            # Get Cohere API key for reranking
            from utils.key_bank import get_keybank
            keybank = get_keybank()
            cohere_key = keybank.get_key("rerank")
            
            semantic_rag = SemanticRAG(
                products=PRODUCTS, 
                persist_dir="/app/chroma_data",
                cohere_api_key=cohere_key  # Enable reranking
            )
            print(f"[STARTUP] SemanticRAG initialized with {len(PRODUCTS)} products", flush=True)
        except Exception as e:
            print(f"[STARTUP] Failed to initialize SemanticRAG: {e}", flush=True)
            import traceback
            traceback.print_exc()
            semantic_rag = None
    else:
        print("[STARTUP] No products loaded, SemanticRAG not initialized", flush=True)

# Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Product(BaseModel):
    SKU: str
    Brand: str
    Family: str
    Name: str
    Description: str
    CPU: Optional[str] = None
    GPU: Optional[str] = None
    Storage: Optional[str] = None
    RAM: Optional[str] = None
    Price: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Laptop Recommendation Chatbot API is running"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify backend is ready."""
    return {
        "status": "ready",
        "products_loaded": len(PRODUCTS) > 0,
        "semantic_rag_ready": semantic_rag is not None
    }

@app.get("/products")
async def get_products():
    """Get the list of available products."""
    return PRODUCTS

@app.get("/chat-history")
async def get_chat_history():
    """Get all chat sessions."""
    chat_logger = get_chat_history_logger(chat_logs_dir)
    return {"sessions": chat_logger.get_all_sessions()}

@app.get("/chat-history/{session_id}")
async def get_session_history(session_id: str):
    """Get a specific chat session."""
    chat_logger = get_chat_history_logger(chat_logs_dir)
    session = chat_logger.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    """Handle a chat message and return a response using the OferGPT instance."""
    try:
        messages = chat_request.messages
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        # Get the latest user message
        user_message = next((msg for msg in reversed(messages) if msg.role == "user"), None)
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # Initialize chatbot
        chatbot = OferGPT()
        
        # Use the global semantic RAG instance (already initialized at startup)
        if semantic_rag:
            chatbot.semantic_rag = semantic_rag
            # Pass the chatbot's API call logger to semantic_rag
            semantic_rag.api_call_logger = chatbot._log_api_call
        
        # Get the chat response
        response_chunks = []
        for chunk in chatbot.chat_stream(
            user_input=user_message.content,
            context={"products": PRODUCTS},
            message_history=[{"role": msg.role, "content": msg.content} for msg in messages[:-1]]
        ):
            response_chunks.append(chunk)
        
        # Combine chunks into a single response
        response = "".join(response_chunks)
        
        return {
            "role": "assistant",
            "content": response,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        error_msg = f"Error processing chat message: {str(e)}"
        print(f"Error: {error_msg}", flush=True)
        raise HTTPException(status_code=500, detail=error_msg)

# WebSocket endpoint for real-time chat
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established", flush=True)
    
    # Initialize the chatbot, chat history logger, and metrics tracker
    chatbot = OferGPT()
    
    # Use the global semantic RAG instance (already initialized at startup)
    if semantic_rag:
        chatbot.semantic_rag = semantic_rag
        # Pass the chatbot's API call logger to semantic_rag
        semantic_rag.api_call_logger = chatbot._log_api_call
    
    chat_logger = get_chat_history_logger(chat_logs_dir)
    metrics_tracker = get_metrics_tracker(chatbot.session_id)
    session_id = chatbot.session_id
    message_history = []
    
    print(f"Chat session started: {session_id}", flush=True)
    print(f"Chat logs directory: {chat_logs_dir}", flush=True)
    print(f"Metrics tracking enabled: {metrics_tracker.enabled}", flush=True)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "chat":
                # Log the raw message for debugging
                print(f"Raw message received: {json.dumps(message)}", flush=True)
                
                # Get the user message - handle both formats
                # Format 1: {"type": "chat", "content": "message"}
                # Format 2: {"type": "chat", "messages": [{"role": "user", "content": "message"}]}
                user_message = message.get("content", "")
                
                if not user_message and "messages" in message:
                    # Extract from messages array (latest user message)
                    messages = message.get("messages", [])
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            user_message = msg.get("content", "")
                            break
                
                print(f"Extracted user_message: '{user_message}'", flush=True)
                
                # Add to message history (no timestamps to save tokens)
                message_history.append({
                    "role": "user",
                    "content": user_message
                })
                
                # Log user message to chat history
                chat_logger.log_message(session_id, "user", user_message)
                
                # Get client IP for logging
                client_ip = websocket.client.host if websocket.client else "unknown"
                print(f"Message from {client_ip}: {user_message[:100] if user_message else '(empty)'}", flush=True)
                
                try:
                    # Track latency from user message to first stream chunk
                    turn_start_time_dt = datetime.utcnow()
                    turn_start_time = time.time()
                    
                    # Stream the response
                    response_chunks = []
                    first_chunk_time_dt = None
                    first_chunk_time = None
                    
                    # Prepare context with products and message history (excluding current message)
                    print(f"PRODUCTS loaded: {len(PRODUCTS)} products available", flush=True)
                    context = {
                        "products": PRODUCTS,
                        "message_history": message_history[:-1]
                    }
                    print(f"Context prepared with {len(context['products'])} products", flush=True)
                    
                    # Get the chat response with full context and explicit message history
                    # chat_stream is now an async generator for non-blocking concurrent connections
                    chat_response = chatbot.chat_stream(
                        user_input=user_message,
                        context=context,
                        message_history=message_history[:-1]
                    )
                    
                    # Process and send chunks using async for (non-blocking)
                    # All LLM calls within chat_stream are now async (ainvoke for intermediate, astream for final)
                    i = 0
                    async for chunk in chat_response:
                        # Capture time of first chunk (when streaming starts)
                        if i == 0:
                            first_chunk_time_dt = datetime.utcnow()
                            first_chunk_time = time.time()
                        
                        if chunk:
                            
                            response_chunks.append(chunk)
                            await websocket.send_json({
                                "type": "chat_chunk",
                                "content": chunk,
                                "is_final": False,
                                "timestamp": datetime.utcnow().isoformat()
                            })
                        i += 1
                    
                    # Calculate latency: from user message to first stream chunk
                    if first_chunk_time_dt is None:
                        # No chunks received, use end time
                        first_chunk_time_dt = datetime.utcnow()
                        first_chunk_time = time.time()
                    
                    latency_ms = (first_chunk_time - turn_start_time) * 1000
                    turn_end_time_dt = first_chunk_time_dt
                    
                    # Debug logging
                    print(f"Turn latency: {latency_ms:.2f}ms (start: {turn_start_time_dt.isoformat()}, end: {turn_end_time_dt.isoformat()})", flush=True)
                    
                    # Add assistant's response to history (no timestamps to save tokens)
                    full_response = "".join(response_chunks)
                    if full_response:
                        message_history.append({
                            "role": "assistant",
                            "content": full_response
                        })
                        
                        # Log assistant response to chat history
                        chat_logger.log_message(session_id, "assistant", full_response)
                        
                        # Gather all data for comprehensive tracking
                        intent = getattr(chatbot, '_last_detected_intent', None)
                        products = []
                        matched_count = 0
                        response_type = None
                        
                        # Get LLM provider and model info
                        llm_provider = os.getenv("LLM_PROVIDER", "cohere").lower()
                        if llm_provider == "cohere":
                            llm_model = os.getenv("COHERE_CHAT_MODEL", "command-a-vision-07-2025")
                        elif llm_provider == "google":
                            llm_model = os.getenv("GOOGLE_MODEL", "gemini-1.5-pro")
                        else:
                            llm_model = "unknown"
                        
                        # Build LLM metadata
                        llm_metadata = {
                            "provider": llm_provider,
                            "model": llm_model,
                        }
                        print(f"[TOKEN_TRACKING] Building LLM metadata - provider={llm_provider}, model={llm_model}", flush=True)
                        
                        # Add token counts if available
                        print(f"[TOKEN_TRACKING] Checking for token counts in chatbot._last_token_counts", flush=True)
                        if hasattr(chatbot, '_last_token_counts'):
                            print(f"[TOKEN_TRACKING] chatbot has _last_token_counts attribute", flush=True)
                            if chatbot._last_token_counts:
                                token_counts = chatbot._last_token_counts
                                llm_metadata["input_tokens"] = token_counts.get('input_tokens', 0)
                                llm_metadata["output_tokens"] = token_counts.get('output_tokens', 0)
                                print(f"[TOKEN_TRACKING] Token counts extracted: input={llm_metadata['input_tokens']}, output={llm_metadata['output_tokens']}", flush=True)
                                print(f"[TOKEN_TRACKING] Added to llm_metadata: {llm_metadata}", flush=True)
                            else:
                                print(f"[TOKEN_TRACKING] chatbot._last_token_counts is None or empty", flush=True)
                        else:
                            print(f"[TOKEN_TRACKING] chatbot does not have _last_token_counts attribute", flush=True)
                        
                        # Add API call counts if available
                        print(f"[API_TRACKING] Checking for API call counts in chatbot._api_counts", flush=True)
                        if hasattr(chatbot, '_api_counts'):
                            print(f"[API_TRACKING] chatbot has _api_counts attribute", flush=True)
                            if chatbot._api_counts:
                                api_counts = chatbot._api_counts
                                total_api_calls = sum(api_counts.values())
                                llm_metadata["total_api_calls"] = total_api_calls
                                # Add breakdown by type
                                api_breakdown = {}
                                for (api_type, key_name), count in api_counts.items():
                                    if count > 0:
                                        api_breakdown[f"{api_type}:{key_name}"] = count
                                llm_metadata["api_calls_breakdown"] = api_breakdown
                                print(f"[API_TRACKING] API call counts extracted: total={total_api_calls}, breakdown={api_breakdown}", flush=True)
                                print(f"[API_TRACKING] Added to llm_metadata: total_api_calls={llm_metadata['total_api_calls']}", flush=True)
                            else:
                                print(f"[API_TRACKING] chatbot._api_counts is empty", flush=True)
                        else:
                            print(f"[API_TRACKING] chatbot does not have _api_counts attribute", flush=True)
                        
                        # Log accumulated preferences if any
                        if hasattr(chatbot, 'accumulated_preferences') and chatbot.accumulated_preferences:
                            chat_logger.update_preferences(session_id, chatbot.accumulated_preferences)
                        
                        # Get product recommendations if any
                        if hasattr(chatbot, 'recommendations_given') and chatbot.recommendations_given:
                            latest_recommendation = chatbot.recommendations_given[-1]
                            chat_logger.log_recommendation(session_id, latest_recommendation)
                            products = latest_recommendation.get("products", [])
                            matched_count = len(products)
                            response_type = "recommendation" if products else "clarification"
                        else:
                            # No recommendations - determine response type from intent and prompt count
                            if intent:
                                if intent in ['greeting', 'chitchat']:
                                    response_type = "greeting"
                                elif intent == 'product_inquiry':
                                    response_type = "product_inquiry"
                                elif intent in ['preferences_given', 'recommendation_request']:
                                    # Check if we're still in the clarification phase
                                    if hasattr(chatbot, 'product_prompt_count') and hasattr(chatbot, 'min_prompts_for_recommendation'):
                                        if chatbot.product_prompt_count < chatbot.min_prompts_for_recommendation:
                                            response_type = "clarification"
                                        else:
                                            response_type = "recommendation"
                                    else:
                                        response_type = "clarification"
                                else:
                                    response_type = intent
                            else:
                                response_type = "general_response"
                        
                        # Get the actual filter_type from the chatbot's last recommendation
                        filter_type = "semantic"  # default
                        if chatbot.recommendations_given:
                            filter_type = chatbot.recommendations_given[-1].get("filter_type", "semantic")
                        
                        # Track turn metrics with ALL data in single Langfuse observation
                        print(f"[TOKEN_TRACKING] Calling metrics_tracker.track_turn() with llm_metadata: {llm_metadata}", flush=True)
                        metrics_tracker.track_turn(
                            user_message=user_message,
                            assistant_response=full_response,
                            latency_ms=latency_ms,
                            start_time=turn_start_time_dt,
                            end_time=turn_end_time_dt,
                            intent=intent,
                            response_type=response_type,
                            products=products,
                            matched_count=matched_count,
                            llm_metadata=llm_metadata,
                            filter_type=filter_type
                        )
                        print(f"[TOKEN_TRACKING] metrics_tracker.track_turn() completed successfully", flush=True)
                        
                        # Also log to local session (for backward compatibility)
                        if products:
                            metrics_tracker.track_recommendation(
                                products=products,
                                matched_count=matched_count,
                                filter_type=filter_type,
                                response_type=response_type
                            )
                        
                        # Send final message
                        await websocket.send_json({
                            "type": "chat_chunk",
                            "content": "",
                            "is_final": True,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    
                    print(f"Finished streaming response to {client_ip} (latency: {latency_ms:.2f}ms)", flush=True)
                
                except Exception as e:
                    error_msg = f"Error processing message: {str(e)}"
                    print(f"Error: {error_msg}", flush=True)
                    error_end_time_dt = datetime.utcnow()
                    metrics_tracker.track_error(
                        error_msg, 
                        error_type="chat_error",
                        start_time=turn_start_time_dt,
                        end_time=error_end_time_dt
                    )
                    try:
                        if hasattr(websocket, 'client_state') and not websocket.client_state.disconnected:
                            await websocket.send_json({
                                "type": "error",
                                "content": "An error occurred while processing your message. Please try again.",
                                "timestamp": datetime.utcnow().isoformat()
                            })
                    except Exception as send_error:
                        print(f"Failed to send error message: {send_error}", flush=True)
                    return

    except WebSocketDisconnect:
        client_ip = websocket.client.host if hasattr(websocket, 'client') and websocket.client else "unknown"
        print(f"WebSocket connection closed by {client_ip}", flush=True)
        # End the chat session
        chat_logger.end_session(session_id)
        metrics_tracker.end_session()
        print(f"Chat session ended: {session_id}", flush=True)
        print(f"Session metrics: {metrics_tracker.get_session_metrics()}", flush=True)
        return
                
    except Exception as e:
        error_msg = f"Unexpected error in WebSocket handler: {str(e)}"
        print(f"Error: {error_msg}", flush=True)
        if hasattr(websocket, 'client_state') and not websocket.client_state.disconnected:
            try:
                await websocket.send_json({
                    "type": "error",
                    "content": "An unexpected error occurred. Please try reconnecting.",
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as send_error:
                print(f"Could not send error message to client: {send_error}", flush=True)
        return


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
