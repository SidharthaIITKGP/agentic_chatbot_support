# visualize_graph.py
"""
Visualize the LLM-powered ReAct agent structure.
Run: uv run python scripts/visualize_graph.py

Note: The new LLM-powered agent uses LangGraph's create_react_agent,
which handles the ReAct loop internally. The graph structure is simpler
but more powerful - all reasoning is done by the LLM!
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.llm_agent import llm_agent

print("🎨 Generating LLM-powered ReAct agent visualization...\n")

# Try to display in IPython/Jupyter if available
try:
    from IPython.display import Image, display
    in_notebook = True
    print("📓 IPython environment detected - will display inline")
except ImportError:
    in_notebook = False

# Generate the graph visualization
try:
    # Get the mermaid PNG from the LLM agent graph
    img_data = llm_agent.get_graph().draw_mermaid_png()
    
    # Save to file
    output_path = PROJECT_ROOT / "agent_graph.png"
    with open(output_path, "wb") as f:
        f.write(img_data)
    
    print(f"✅ Graph visualization saved to {output_path}")
    
    # Display inline if in notebook
    if in_notebook:
        print("\n📊 Displaying graph:")
        display(Image(img_data))
    else:
        print("\n💡 To view the graph:")
        print(f"   - Open: {output_path}")
        print("   - Or in IPython/Jupyter:")
        print("     from IPython.display import Image, display")
        print("     display(Image('agent_graph.png'))")
    
    print("\n🔄 LLM-Powered ReAct Pattern:")
    print("   User Query → LLM Reasoning → Tool Selection → Tool Execution")
    print("                     ↑                                ↓")
    print("                     └────────────────────────────────┘")
    print("   LLM decides when to stop (no hardcoded iteration limit!)")
    
except ImportError as e:
    print("⚠️  Missing dependencies for PNG generation")
    print("   Install with: pip install pygraphviz")
    print("\n📝 Showing ASCII representation instead:\n")
    try:
        print(llm_agent.get_graph().draw_ascii())
    except:
        print(llm_agent.get_graph())
        
except Exception as e:
    print(f"⚠️  Could not generate PNG: {e}")
    print("\n📝 Showing ASCII representation instead:\n")
    try:
        print(llm_agent.get_graph().draw_ascii())
    except:
        print("\n🔄 LLM-Powered ReAct Agent Structure:")
        print("  ┌─────────────────────────────────┐")
        print("  │  LLM ReAct Agent                │")
        print("  │  (LangGraph create_react_agent) │")
        print("  │                                 │")
        print("  │  LLM thinks → decides action    │")
        print("  │       ↓                         │")
        print("  │  Tool execution or finish       │")
        print("  │       ↓                         │")
        print("  │  LLM observes result            │")
        print("  │       ↓                         │")
        print("  │  Loop until LLM decides done    │")
        print("  └─────────────────────────────────┘")
