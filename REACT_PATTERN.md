# ReAct Pattern Implementation

## Overview

This customer support agent now uses the **ReAct (Reasoning and Acting)** pattern, which combines natural language reasoning with actions to solve tasks iteratively.

## What is ReAct?

ReAct is an AI agent architecture that alternates between three steps:
1. **Thought**: The agent reasons about what to do next
2. **Action**: The agent takes an action (calls a tool, queries knowledge base, or finishes)
3. **Observation**: The agent observes the result of the action

This cycle repeats until the agent has enough information to answer the user's query.

## Architecture

### State (AgentState)
The state includes ReAct-specific fields:
- `thought`: Current reasoning step (what the agent is thinking)
- `action`: Action to take ("call_tool", "call_rag", "ask_for_slot", "finish")
- `action_input`: Parameters for the action
- `observation`: Result of the last action
- `scratchpad`: Accumulated thoughts/actions/observations for context
- `iteration`: Current iteration count (max: 5)

### Graph Flow

```
classify_intent → reasoning → action → [loop or compose]
                      ↑          ↓
                      └─────────┘ (ReAct loop)
```

1. **classify_intent_node**: Initial classification of user intent
2. **reasoning_node**: Generates thoughts and decides next action
3. **action_node**: Executes the decided action
4. **Loop back or continue**:
   - If `action == "ask_for_slot"`: END (need user input)
   - If `action == "finish"`: → compose_node
   - Otherwise: → reasoning_node (continue iteration)
5. **compose_node**: Generates final answer

### Iteration Limit
Maximum 5 iterations to prevent infinite loops. If exceeded, the agent proceeds to compose the final answer.

## Example Traces

### Order Status Query
```
Query: Where is my order 98762?

Iteration 1:
  Thought: I have order ID 98762. I should call the order_status tool to get current status.
  Action: call_tool
  Observation: Retrieved order status: N/A

Iteration 2:
  Thought: I have the tool result. Let me get relevant policy information to provide complete context.
  Action: call_rag
  Observation: Retrieved 2 relevant policy documents.

Iteration 3:
  Thought: I have all the information needed. Time to compose the final answer.
  Action: finish
  Observation: Ready to compose final answer.

Final Answer: Order 98762 is currently: Out for delivery...
```

### Policy Query
```
Query: What is your return policy?

Iteration 1:
  Thought: This is a policy question. I should search the knowledge base for relevant information.
  Action: call_rag
  Observation: Retrieved 3 relevant policy documents.

Iteration 2:
  Thought: I have all the information needed. Time to compose the final answer.
  Action: finish
  Observation: Ready to compose final answer.

Final Answer: [Policy information]
```

## Benefits

1. **Transparency**: The scratchpad shows the agent's reasoning process
2. **Flexibility**: Agent can adapt its approach based on observations
3. **Error Recovery**: Can adjust strategy if an action fails
4. **Iterative Refinement**: Can gather information incrementally
5. **Explainability**: Clear thought process for debugging and user trust

## Key Differences from Previous Implementation

| Aspect | Before | ReAct |
|--------|--------|-------|
| Flow | Linear pipeline | Iterative loop |
| Decision Making | Hardcoded routing | Reasoning node decides |
| Transparency | Hidden logic | Explicit thoughts |
| Flexibility | Fixed sequence | Dynamic adaptation |
| Debugging | Black box | Full trace available |

## Testing

Run the ReAct test suite to see reasoning traces:
```bash
uv run python test_react.py
```

This will show the complete thought process for multiple query types.

## Configuration

- `MAX_ITERATIONS`: Set in `src/agent/agent_graph.py` (default: 5)
- Actions available: "ask_for_slot", "call_tool", "call_rag", "finish"
- Routing logic: Defined in `route_after_action()` function

## Memory Integration

The ReAct agent fully supports persistent memory:
- Memory is seeded before the first reasoning iteration
- Slots from previous conversations are carried forward
- Intent correction based on conversation history works seamlessly

```python
from src.agent.agent_graph import run_agent_with_memory

# Session-based memory
result = run_agent_with_memory("98762", session_id="user123")
```

## Graph Visualization

View the ReAct graph structure:
```bash
uv run python scripts/visualize_graph.py
```

This generates `agent_graph.png` showing the ReAct loop with:
- classify_intent node
- reasoning node (ReAct Thought)
- action node (ReAct Action)
- compose node (Final Answer)
- Conditional edges for the loop

## Future Enhancements

Possible improvements:
1. **Adaptive iteration limits**: Adjust based on query complexity
2. **Reflection**: Agent critiques its own reasoning
3. **Planning**: Multi-step plan generation before acting
4. **Tool selection**: Reasoning about which tool to use
5. **Parallel actions**: Execute multiple actions simultaneously
