# visualize_graph.py
"""
Visualize the LangGraph workflow structure.
Run: uv run python visualize_graph.py
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent_graph import agent_graph

# Generate and save the graph visualization
try:
    from IPython.display import Image, display
    
    # Generate the graph image
    img_data = agent_graph.get_graph().draw_mermaid_png()
    
    # Save to file
    with open("agent_graph.png", "wb") as f:
        f.write(img_data)
    
    print("✅ Graph visualization saved to agent_graph.png")
    print("\nYou can also view it with:")
    print("  from IPython.display import Image, display")
    print("  display(Image('agent_graph.png'))")
    
except ImportError:
    print("To generate graph visualization, install: uv add pygraphviz")
    print("Or use the ASCII representation:")
    print("\n" + agent_graph.get_graph().draw_ascii())
except Exception as e:
    print(f"Could not generate PNG. Showing ASCII representation instead:\n")
    try:
        print(agent_graph.get_graph().draw_ascii())
    except:
        print("Graph structure:")
        print("  classify_intent -> ask_for_slot")
        print("  ask_for_slot -> (conditional)")
        print("    -> call_tool (if continuing)")
        print("    -> END (if final_answer set)")
        print("  call_tool -> call_rag")
        print("  call_rag -> compose")
        print("  compose -> END")
