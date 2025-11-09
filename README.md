# Yod.ai 🤖💻

An intelligent AI-powered chatbot for laptop recommendations that understands both technical specifications and semantic preferences using advanced RAG (Retrieval-Augmented Generation) and LLM capabilities.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![React](https://img.shields.io/badge/react-19.2.0-blue.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.115.0-green.svg)

## 🌟 Features

### Core Capabilities
- **Semantic Understanding**: Uses ChromaDB embeddings to understand abstract user needs (e.g., "deep learning", "video editing")
- **Multi-Provider LLM Support**: Switch between Cohere and Google Gemini models
- **Hybrid Filtering**: Combines semantic search with hard spec filtering (CPU, GPU, RAM, storage, price)
- **Cohere Reranking**: Enhances semantic search results with Cohere's rerank-english-v3.0
- **Persistent Embeddings**: ChromaDB stores embeddings on disk for fast startup
- **Real-time Streaming**: WebSocket-based chat for instant responses
- **Preference Accumulation**: Remembers user preferences throughout the conversation
- **Smart Questioning**: Never re-asks for specifications already provided

### Tracking & Analytics
- **Chat History Logging**: Persistent JSON logs of all conversations
- **Metrics Tracking**: Langfuse integration for:
  - Time-to-first-chunk (TTFC) latency
  - Token usage (input/output)
  - Turn-level metrics
  - Session-level analytics
- **Comprehensive Logging**: Debug-level logs at every stage (LLM provider, chatbot, metrics)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Frontend (React)                      │
│  - Material-UI components                                   │
│  - WebSocket client for real-time chat                      │
│  - Chat history & preference tracking                       │
└──────────────────┬──────────────────────────────────────────┘
                   │ WebSocket + REST API
┌──────────────────▼──────────────────────────────────────────┐
│                    Backend (FastAPI)                        │
│  - WebSocket handler (/ws)                                  │
│  - REST endpoints (/chat, /products, /health)               │
│  - Chat history logger                                      │
│  - Metrics tracker (Langfuse)                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│                     Chatbot (OferGPT)                       │
│  - Preference extraction (LLM)                              │
│  - Semantic context accumulation                            │
│  - Response generation                                      │
└─────┬────────────────────────────────┬─────────────────────┘
      │                                │
┌─────▼─────────────────┐   ┌──────────▼──────────────────────┐
│   LLM Provider Layer  │   │   Semantic RAG (ChromaDB)      │
│  - Cohere             │   │  - Embeddings (all-MiniLM-L6-v2)│
│  - Google Gemini      │   │  - Cohere reranking             │
│  - Token tracking     │   │  - Persistent storage           │
└───────────────────────┘   └─────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- API Keys:
  - Cohere API key (for LLM and reranking)
  - Google API key (optional, for Gemini models)
  - Langfuse API keys (optional, for metrics tracking)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Yod.ai
   ```

2. **Create `.env` file**
   ```bash
   # Copy the example and fill in your API keys
   cp .env.example .env
   ```

3. **Configure environment variables**
   Edit `.env` with your API keys:
   ```env
   # LLM Provider (cohere or google)
   LLM_PROVIDER=cohere
   
   # Cohere Configuration
   COHERE_API_KEY=your_cohere_api_key_here
   COHERE_CHAT_MODEL=command-a-vision-07-2025
   
   # Google AI Configuration (optional)
   GOOGLE_API_KEY=your_google_api_key_here
   GOOGLE_MODEL=gemini-2.0-flash
   
   # Langfuse Metrics (optional)
   LANGFUSE_SECRET_KEY=your_langfuse_secret_key
   LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

4. **Start the application**
   ```bash
   docker-compose up --build
   ```

5. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 🔧 Configuration

### LLM Provider Setup

#### Cohere (Recommended)
- Fast streaming responses
- Built-in reranking support
- Stable token tracking

```env
LLM_PROVIDER=cohere
COHERE_API_KEY=your_key_here
COHERE_CHAT_MODEL=command-a-vision-07-2025
```

#### Google Gemini
- Alternative LLM option
- Uses `gemini-2.0-flash` by default (stable)

```env
LLM_PROVIDER=google
GOOGLE_API_KEY=your_key_here
GOOGLE_MODEL=gemini-2.0-flash
```

### Key Bank System

The project uses a key bank system for managing multiple API keys:
- Keys are stored in environment variables
- Automatic key rotation for load balancing
- Supports multiple keys for the same service

### Langfuse Metrics (Optional)

Enable comprehensive tracking:
```env
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

**Metrics tracked:**
- Time-to-first-chunk (TTFC) latency
- Input/output token counts
- Turn-level metadata (intent, response type)
- Product recommendations
- Session-level analytics

## 📁 Project Structure

```
Yod.ai/
├── backend/
│   ├── chatbot.py              # Core chatbot logic (OferGPT)
│   ├── main.py                 # FastAPI application & WebSocket handler
│   ├── utils/
│   │   ├── chat_history_logger.py    # Persistent chat logging
│   │   ├── key_bank.py              # API key management
│   │   ├── laptop_filter.py         # Product filtering logic
│   │   ├── llm_provider.py          # Multi-provider LLM abstraction
│   │   ├── metrics_tracker.py       # Langfuse integration
│   │   └── semantic_rag.py          # ChromaDB RAG system
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx    # Main chat UI
│   │   │   ├── LoadingScreen.tsx    # Yoda loading screen
│   │   │   └── ProductCard.tsx      # Product display
│   │   ├── contexts/
│   │   │   └── ChatContext.tsx      # Chat state management
│   │   ├── App.tsx
│   │   └── index.tsx
│   ├── Dockerfile
│   └── package.json
├── products.json               # Laptop product database
├── docker-compose.yml
└── README.md
```

## 🎯 How It Works

### 1. Preference Extraction
When a user sends a message, the chatbot extracts:
- **Hard specs**: vendor, CPU, GPU, RAM, storage, max_price
- **Semantic preferences**: battery_life, use_case, portability, display

### 2. Semantic Search
- User preferences are converted to embeddings
- ChromaDB searches for semantically similar laptops
- Results are reranked using Cohere's rerank model
- Hard spec filters are applied via ChromaDB metadata

### 3. Preference Accumulation
- All preferences are accumulated across conversation turns
- New preferences override old ones
- The chatbot never re-asks for already-provided specs

### 4. Response Generation
- LLM generates contextual responses
- Product recommendations include full specifications
- Smart follow-up questions for missing information

### Query Flow Example

```
Turn 1: "I need good battery life with 16GB RAM"
→ Extracted: {ram: 16GB, battery_life: good}
→ Semantic query: "I need a laptop with battery life: good"
→ Filters: RAM ≥ 16GB

Turn 2: "I need it for deep learning"
→ Accumulated: {ram: 16GB, battery_life: good, use_case: deep learning}
→ Semantic query: "battery life: good, use_case: deep learning"
→ Filters: RAM ≥ 16GB

Turn 3: "under $3000"
→ Accumulated: {ram: 16GB, battery_life: good, use_case: deep learning, max_price: 3000}
→ Applies all filters and returns matching laptops
```

## 🐳 Docker Deployment

### Development
```bash
docker-compose up --build
```

### Production
The project includes Railway-specific Dockerfiles for deployment:
- `backend/Dockerfile.railway`
- `frontend/Dockerfile.railway`

### Data Persistence
Docker volumes ensure data persists across restarts:
- `chroma_data`: ChromaDB embeddings
- `chat_logs`: Conversation history

## 🔍 API Endpoints

### REST API
- `GET /` - Health check
- `GET /api/health` - Backend readiness check
- `GET /products` - List all products
- `POST /chat` - Single chat message
- `GET /chat-history` - All chat sessions
- `GET /chat-history/{session_id}` - Specific session

### WebSocket
- `WS /ws` - Real-time chat streaming

## 📊 Logging & Debugging

### Log Levels
- **DEBUG**: Detailed token extraction, chunk processing
- **INFO**: High-level operations, metrics tracking
- **WARNING**: Missing data, fallbacks
- **ERROR**: Exceptions and failures

### Log Prefixes
Filter logs by component:
- `[COHERE_*]` - Cohere provider logs
- `[GOOGLE_*]` - Google provider logs
- `[CHAT_STREAM]` - Chatbot streaming logs
- `[TOKEN_TRACKING]` - Token tracking logs
- `[METRICS_*]` - Metrics and Langfuse logs
- `[STARTUP]` - Application initialization

### Example Debug Flow
```
[STARTUP] Initializing SemanticRAG system...
[STARTUP] SemanticRAG initialized with 31 products
[CHAT_STREAM] Starting to stream response from LLM
[COHERE_STREAM] Stream complete. Total chunks: 45
[TOKEN_TRACKING] Token counts extracted: input=234, output=567
[METRICS_LANGFUSE] Generation observation created successfully
```

## 🧪 Testing

### Manual Testing
1. Start the application: `docker-compose up`
2. Open http://localhost:3000
3. Try sample queries:
   - "I need a laptop for video editing under $2000"
   - "Show me gaming laptops with RTX 4080"
   - "I want good battery life for travel"

### Health Check
```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "ready",
  "products_loaded": true,
  "semantic_rag_ready": true
}
```

## 🙏 Acknowledgments

- **ChromaDB** - Vector database for embeddings
- **Cohere** - LLM and reranking services
- **Google Gemini** - Alternative LLM provider
- **Langfuse** - LLM metrics and observability
- **FastAPI** - Modern Python web framework
- **React** - Frontend framework
