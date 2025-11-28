"""
Test script to demonstrate ReAct reasoning process.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent_graph import agent_graph

def test_react_with_trace(query: str):
    """Test the ReAct agent and show the reasoning trace."""
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}\n")
    
    # Create initial state
    initial_state = {
        "user_query": query,
        "intent": None,
        "slots": {},
        "thought": None,
        "action": None,
        "action_input": None,
        "observation": None,
        "tool_response": None,
        "rag_results": [],
        "final_answer": None,
        "history": [],
        "iteration": 0,
        "scratchpad": "",
        "errors": []
    }
    
    # Run the graph
    result = agent_graph.invoke(initial_state)
    
    # Display ReAct trace
    print("REASONING TRACE:")
    print("-" * 70)
    scratchpad = result.get("scratchpad", "")
    if scratchpad:
        print(scratchpad)
    else:
        print("(No reasoning trace available)")
    
    print("\n" + "=" * 70)
    print("FINAL ANSWER:")
    print("-" * 70)
    print(result.get("final_answer", "No answer generated"))
    print("=" * 70)
    
    # Display metadata
    print("\nMETADATA:")
    print(f"  Intent: {result.get('intent')}")
    print(f"  Slots: {result.get('slots')}")
    print(f"  Iterations: {result.get('iteration')}")
    print(f"  Tool called: {result.get('tool_response') is not None}")
    print(f"  RAG results: {len(result.get('rag_results', []))}")
    print()

if __name__ == "__main__":
    # Test cases
    test_cases = [
        "Where is my order 98762?",
        "What is your return policy?",
        "Is product P456 available?",
    ]
    
    for test_query in test_cases:
        test_react_with_trace(test_query)
