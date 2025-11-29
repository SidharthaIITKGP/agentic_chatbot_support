# Agentic Customer Support Chatbot

A sophisticated customer support agent powered by **LangGraph ReAct pattern with LLM reasoning**, combining intelligent LLM-driven decision making, tool execution, and RAG (Retrieval-Augmented Generation) for intelligent query handling.

## 🌟 Features

### True LLM-Powered ReAct Agent
- **LLM Reasoning**: Language model thinks through problems autonomously
- **Tool Selection**: LLM decides which tools to use and when
- **Adaptive Learning**: Observes results and adapts strategy dynamically
- **Natural Conversation**: No hardcoded rules - pure AI reasoning
- **Transparent**: Full reasoning traces show LLM thought process

### Capabilities
- ✅ Order status tracking
- ✅ Refund status inquiries
- ✅ Product availability checks
- ✅ Policy questions (return, refund, delivery, charges, etc.)
- ✅ Context-aware conversation memory
- ✅ Multi-step reasoning and tool orchestration
- ✅ Intelligent query understanding

### Technology Stack
- **LangGraph**: ReAct agent orchestration with `create_react_agent`
- **LangChain**: LLM integration and tool framework
- **Groq**: Fast inference with `llama-3.1-8b-instant` model
- **RAG**: FAISS vector store with HuggingFace embeddings (all-MiniLM-L6-v2)
- **Streamlit**: Clean chat interface with optional reasoning display
- **UV**: Modern Python package management

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/SidharthaIITKGP/agentic_chatbot_support.git
cd "agentic_chatbot_support"

# Install dependencies (using uv)
uv sync
```

### Environment Setup

Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key from: https://console.groq.com/

### Setup Policy Documents

Ingest policy documents into the vector store:

```bash
uv run python src/rag/ingest_policies.py
```

### Run the Agent

**CLI Mode**:
```bash
uv run python -m src.agent.llm_agent "Where is my order 98762?"
```

**Streamlit UI** (Recommended):
```bash
uv run streamlit run src/ui/streamlit_app.py --server.port 8503
```

Open http://localhost:8503 in your browser.

**Enable reasoning traces** by checking "Show ReAct Reasoning" in the sidebar!

## 📊 LLM ReAct in Action

```
User: Where is my order 98762 and what if it's delayed?

LLM Thought Process:
  💭 "I need to check the order status first. Let me use the 
      get_order_status tool with ID 98762."
      
  🎬 Action: get_order_status_tool(order_id="98762")
  
  👁️ Observation: {order_id: 98762, status: "out for delivery", 
                   expected_delivery: "2025-01-24"}
  
  💭 "Good, the order is on the way. Now they asked about delays,
      so I should search the policy documents for delivery delay 
      information."
      
  🎬 Action: search_policy_documents_tool(query="delivery delay policy")
  
  👁️ Observation: Retrieved policy: "If delayed beyond 3 days of 
                   expected delivery, customers eligible for delivery 
                   fee refunds..."
  
  💭 "Perfect! I have both the order status and the delay policy.
      I can now compose a complete answer."

Final Answer: Your order 98762 is currently out for delivery, 
expected on January 24, 2025. If it's delayed beyond 3 days of 
the expected date, you'll be eligible for a delivery fee refund 
according to our delivery delay policy...
```

## 🧪 Testing

### Comprehensive LLM Agent Tests
```bash
uv run python test_llm_agent.py
```

Tests all capabilities:
- Order status queries (with/without ID)
- Refund status checks
- Product availability
- Policy questions
- Complex multi-step reasoning

## 📁 Project Structure

```
.
├── .env                         # API keys (GROQ_API_KEY)
├── src/
│   ├── agent/
│   │   ├── llm_agent.py        # LLM-powered ReAct agent (main)
│   │   ├── llm_tools.py        # LangChain tool wrappers
│   │   └── memory.py           # Persistent conversation memory
│   ├── rag/
│   │   ├── embeddings.py       # HuggingFace embeddings
│   │   ├── ingest_policies.py  # Policy document ingestion
│   │   └── retriever.py        # FAISS vector store retriever
│   ├── tools/
│   │   ├── tools.py            # Order, refund, inventory APIs
│   │   ├── orders.json         # Mock order data
│   │   ├── refunds.json        # Mock refund data
│   │   └── inventory.json      # Mock inventory data
│   └── ui/
│       └── streamlit_app.py    # Chat interface with reasoning display
├── Policy/                      # Policy documents for RAG
│   ├── return_policy.txt
│   ├── refund_policy.txt
│   ├── delivery_delay_policy.txt
│   ├── payment_policy.txt
│   ├── charges_policy.txt
│   ├── cancellation_policy.txt
│   └── replacement_damage_policy.txt
├── test_llm_agent.py            # Comprehensive test suite
└── README.md
```

## 🎯 LLM Agent Architecture

```
User Query
    ↓
┌─────────────────────────────────┐
│  LLM ReAct Agent                │
│  (LangGraph create_react_agent) │
│                                 │
│  LLM thinks → decides action    │
│       ↓                         │
│  Tool execution or finish       │
│       ↓                         │
│  LLM observes result            │
│       ↓                         │
│  Loop until LLM decides done    │
└─────────────────────────────────┘
    ↓
Final Answer
    ↓
Return to User
```

### Available Tools
1. **get_order_status_tool** - Track order status and delivery info
2. **get_refund_status_tool** - Check refund processing status
3. **check_product_availability_tool** - Verify product stock
4. **search_policy_documents_tool** - RAG-powered policy search

### Key Advantages Over Rule-Based Systems
- ✅ No hardcoded if/else logic
- ✅ LLM decides tool usage dynamically
- ✅ Handles novel queries without code changes
- ✅ Natural multi-step reasoning
- ✅ Adapts to conversation context
- ✅ True AI-powered decision making

## 🔧 Configuration

### LLM Model
Edit `src/agent/llm_agent.py`:
```python
MODEL_NAME = "llama-3.1-8b-instant"  # Groq model
TEMPERATURE = 0.1  # Lower = more focused
```

### RAG Parameters
Edit `src/rag/retriever.py`:
```python
retrieve_policy(query, fetch_k=10, top_k=3, alpha=0.85)
```

### Memory Settings
Memory stored in `src/logs/agent_memory_*.json` with LangGraph checkpointing.

## 📚 Documentation

- **[README.md](README.md)**: This file - setup and usage
- **[test_llm_agent.py](test_llm_agent.py)**: Comprehensive test examples

## 🔍 Use Cases

### Order Tracking
```
User: Where is my order?
LLM: I need the order ID to look that up. Could you provide it?
User: 98762
LLM: [Uses get_order_status_tool] Order 98762 is out for delivery...
```

### Policy Questions
```
User: What is your return policy?
LLM: [Uses search_policy_documents_tool with query "return policy"]
     Based on our return policy, you can return items within 30 days...
```

### Complex Multi-Step Queries
```
User: Check order 98762 and tell me about delivery delays
LLM: [Step 1: Uses get_order_status_tool]
     [Step 2: Uses search_policy_documents_tool for delay policy]
     Your order is out for delivery. If delayed beyond 3 days...
```

## 🤝 Contributing

This is an educational/demonstration project showcasing LLM-powered ReAct pattern with LangGraph.

## 📝 License

MIT License

## 🙏 Acknowledgments

- LangGraph for the graph-based orchestration framework
- LangChain for LLM integration and tool framework
- Groq for fast inference
- The ReAct paper: [Yao et al., 2023](https://arxiv.org/abs/2210.03629)

---

**Built with ❤️ using LLM-Powered LangGraph ReAct Pattern**
