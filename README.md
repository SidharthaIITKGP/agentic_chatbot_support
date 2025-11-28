# Agentic Customer Support Chatbot

A sophisticated customer support agent powered by **LangGraph ReAct pattern**, combining reasoning, tool execution, and RAG (Retrieval-Augmented Generation) for intelligent query handling.

## 🌟 Features

### ReAct Agent Architecture
- **Reasoning**: Agent thinks through problems step-by-step
- **Acting**: Executes tools and queries knowledge base
- **Observing**: Learns from results and adapts
- **Transparent**: Full reasoning traces available

### Capabilities
- ✅ Order status tracking
- ✅ Refund status inquiries
- ✅ Product availability checks
- ✅ Policy questions (return, refund, delivery, charges, etc.)
- ✅ Context-aware conversation memory
- ✅ Slot filling for missing information

### Technology Stack
- **LangGraph**: ReAct agent orchestration with iterative reasoning
- **LangChain**: RAG implementation with FAISS vector store
- **HuggingFace**: Sentence transformers (all-MiniLM-L6-v2)
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

### Setup Policy Documents

Ingest policy documents into the vector store:

```bash
uv run python src/rag/ingest_policies.py
```

### Run the Agent

**CLI Mode**:
```bash
uv run python -m src.agent.agent_graph "Where is my order 98762?"
```

**Streamlit UI** (Recommended):
```bash
uv run streamlit run src/ui/streamlit_app.py --server.port 8503
```

Open http://localhost:8503 in your browser.

**Enable reasoning traces** by checking "Show ReAct Reasoning" in the sidebar!

## 📊 ReAct in Action

```
User: Where is my order 98762?

Iteration 1:
  💭 Thought: I have order ID 98762. I should call the order_status 
              tool to get current status.
  🎬 Action: call_tool
  👁️ Observation: Retrieved order status: Out for delivery

Iteration 2:
  💭 Thought: I have the tool result. Let me get relevant policy 
              information to provide complete context.
  🎬 Action: call_rag
  👁️ Observation: Retrieved 2 relevant policy documents.

Iteration 3:
  💭 Thought: I have all the information needed. Time to compose the 
              final answer.
  🎬 Action: finish

Final Answer: Order 98762 is currently: Out for delivery. 
Expected delivery: 2025-01-24...
```

## 🧪 Testing

### Basic Tests
```bash
uv run python test.py
```

### ReAct Reasoning Tests
```bash
uv run python test_react.py
```

Shows full reasoning traces for:
- Order status queries
- Policy questions
- Product availability

### Visualize Graph
```bash
uv run python scripts/visualize_graph.py
```

Generates `agent_graph.png` showing the ReAct loop.

## 📁 Project Structure

```
.
├── src/
│   ├── agent/
│   │   ├── agent_graph.py      # ReAct LangGraph orchestrator
│   │   ├── state.py            # AgentState with ReAct fields
│   │   ├── nodes.py            # reasoning_node, action_node, compose_node
│   │   ├── intent_classifier.py
│   │   ├── composer.py
│   │   └── memory.py           # Persistent conversation memory
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── ingest_policies.py
│   │   └── retriever.py
│   ├── tools/
│   │   ├── tools.py            # Order, refund, inventory APIs
│   │   ├── orders.json
│   │   ├── refunds.json
│   │   └── inventory.json
│   └── ui/
│       └── streamlit_app.py    # Chat interface with reasoning display
├── Policy/                      # Policy documents for RAG
│   ├── return_policy.txt
│   ├── refund_policy.txt
│   ├── delivery_delay_policy.txt
│   └── ...
├── test.py                      # Comprehensive test suite
├── test_react.py                # ReAct reasoning demonstrations
└── scripts/
    └── visualize_graph.py       # Graph visualization
```

## 🎯 Agent Flow

```
User Query
    ↓
Classify Intent
    ↓
┌───────────────────┐
│  ReAct Loop       │
│  (max 5 iterations)│
│                   │
│  Reasoning Node   │ ──→ Decide action
│       ↓          │
│  Action Node     │ ──→ Execute action
│       ↓          │
│  Observation     │ ──→ Record result
│       ↓          │
│  Loop or Finish? │
└───────────────────┘
    ↓
Compose Answer
    ↓
Return to User
```

## 🔧 Configuration

### Iteration Limit
Edit `src/agent/agent_graph.py`:
```python
MAX_ITERATIONS = 5  # Adjust as needed
```

### RAG Parameters
Edit `src/agent/nodes.py`:
```python
retrieve_policy(query, fetch_k=10, top_k=3, alpha=0.85)
```

### Memory Settings
Memory stored in `src/logs/agent_memory_*.json` (last 20 messages).

## 📚 Documentation

- **[REACT_QUICKSTART.md](REACT_QUICKSTART.md)**: Quick guide to ReAct features
- **[REACT_PATTERN.md](REACT_PATTERN.md)**: Detailed ReAct architecture
- **[LANGGRAPH_MIGRATION.md](LANGGRAPH_MIGRATION.md)**: Migration from manual pipeline

## 🔍 Use Cases

### Order Tracking
```
User: Where is my order?
Agent: I can check that for you — could you share the order ID?
User: 98762
Agent: Order 98762 is currently: Out for delivery...
```

### Policy Questions
```
User: What is your return policy?
Agent: [ReAct reasoning: Query RAG → Retrieve policy docs → Compose answer]
```

### Product Availability
```
User: Is product P123 in stock?
Agent: [ReAct: Call inventory tool → Get policy context → Answer]
```

## 🤝 Contributing

This is an educational/demonstration project showcasing ReAct pattern implementation with LangGraph.

## 📝 License

MIT License

## 🙏 Acknowledgments

- LangGraph for the graph-based orchestration framework
- LangChain for RAG components
- The ReAct paper: [Yao et al., 2023](https://arxiv.org/abs/2210.03629)

---

**Built with ❤️ using LangGraph ReAct Pattern**
