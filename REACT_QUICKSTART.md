# ReAct Agent - Quick Start Guide

## What Changed?

Your customer support agent now uses the **ReAct (Reasoning and Acting)** pattern, making it:
- ✅ **Transparent**: See exactly how the agent thinks
- ✅ **Adaptive**: Agent adjusts its approach based on what it learns
- ✅ **Iterative**: Loops through reasoning until it has enough information
- ✅ **Explainable**: Full reasoning trace for debugging

## ReAct Flow

```
User Query → Classify Intent → Reasoning Loop → Final Answer
                                    ↓
                          ┌─────────┴──────────┐
                          ↓                    ↑
                    Reason (Think) ────→ Act (Do) ───→ Observe
                          ↑                    ↓
                          └────────────────────┘
                                (loop until done)
```

## Example: Order Status Query

**Input**: "Where is my order 98762?"

**ReAct Trace**:
```
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
  💭 Thought: I have all the information needed. Time to compose 
              the final answer.
  🎬 Action: finish
  👁️ Observation: Ready to compose final answer.
```

**Output**: "Order 98762 is currently: Out for delivery. Expected delivery: 2025-01-24..."

## Running the Agent

### CLI (shows final answer only)
```bash
uv run python -m src.agent.agent_graph "Where is my order 98762?"
```

### Test with Reasoning Traces
```bash
uv run python test_react.py
```

This runs 3 test cases and shows the complete thought process:
1. Order status query
2. Policy question
3. Product availability

### Streamlit UI (with optional reasoning display)
```bash
uv run streamlit run src/ui/streamlit_app.py --server.port 8503
```

**New feature**: Check "Show ReAct Reasoning" in the sidebar to see the agent's thought process in real-time!

### Full Test Suite
```bash
uv run python test.py
```

## Key Features

### 1. Adaptive Decision Making
The agent decides at each step:
- Need more info from user? → ask_for_slot
- Need to call a tool? → call_tool
- Need policy info? → call_rag
- Have everything? → finish

### 2. Safety with Iteration Limits
Maximum 5 iterations prevent infinite loops. If the agent gets stuck, it generates the best answer it can.

### 3. Full Transparency
The `scratchpad` field accumulates all thoughts, actions, and observations:
```python
result = agent_graph.invoke(initial_state)
print(result["scratchpad"])
```

### 4. Memory Integration
ReAct works seamlessly with conversation memory:
```python
# First message
run_agent_with_memory("Where is my order?", session_id="user123")
# → "Could you share the order ID?"

# Follow-up (remembers context)
run_agent_with_memory("98762", session_id="user123")
# → "Order 98762 is currently: Out for delivery..."
```

## Configuration

Edit `src/agent/agent_graph.py`:

```python
# Increase iteration limit for complex queries
MAX_ITERATIONS = 10

# Or decrease for faster responses
MAX_ITERATIONS = 3
```

## Debugging

### View Graph Structure
```bash
uv run python scripts/visualize_graph.py
```

Opens `agent_graph.png` showing:
- classify_intent → reasoning → action (loop) → compose

### Enable Verbose Logging
```python
from src.agent.agent_graph import agent_graph

result = agent_graph.invoke(initial_state)

# Print iteration count
print(f"Completed in {result['iteration']} iterations")

# Print each thought
for line in result['scratchpad'].split('\n'):
    if line.strip():
        print(line)
```

## Architecture Overview

### Nodes
1. **classify_intent_node**: Understands user intent
2. **reasoning_node**: Generates thoughts, decides actions
3. **action_node**: Executes actions (tools/RAG/finish)
4. **compose_node**: Generates final answer

### State Fields (ReAct-specific)
- `thought`: Current reasoning step
- `action`: What to do next
- `action_input`: Parameters for action
- `observation`: Result of action
- `scratchpad`: Full reasoning history
- `iteration`: Current loop count

### Actions Available
- `"ask_for_slot"`: Request missing information → END
- `"call_tool"`: Execute order/refund/inventory tool → continue
- `"call_rag"`: Query policy knowledge base → continue
- `"finish"`: Done reasoning → compose answer

## Migration from Previous Version

✅ **Backward compatible**: All existing code works
✅ **Same outputs**: Final answers unchanged
✅ **New capability**: Can now see reasoning process

**No breaking changes!** The agent just got smarter and more transparent.

## Performance

- **Average iterations**: 2-3 per query
- **Order status**: 3 iterations (classify → tool → rag → finish)
- **Policy questions**: 2 iterations (classify → rag → finish)
- **Simple queries**: 1-2 iterations

## Documentation

- **Full guide**: `REACT_PATTERN.md`
- **Migration docs**: `LANGGRAPH_MIGRATION.md`
- **README**: `README.md`

## Support

If you see unexpected behavior:
1. Check iteration count (>5 means it hit the limit)
2. Review scratchpad for reasoning errors
3. Clear memory: Click "Clear Chat + Memory" in Streamlit

Enjoy your ReAct-powered agent! 🚀
