# visualize_graph.py
"""
Visualize the LLM-powered ReAct agent graph.
Run: python scripts/visualize_graph.py
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.llm_agent import llm_agent

print("🎨 Generating LangGraph Agent Visualization...\n")
print("=" * 70)

# Get the graph
graph = llm_agent.get_graph()

# Generate PNG
try:
    print("Generating graph image...")
    img_data = graph.draw_mermaid_png()
    
    output_path = PROJECT_ROOT / "agent_graph.png"
    with open(output_path, "wb") as f:
        f.write(img_data)
    
    print(f"✅ Graph saved to: {output_path}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTrying ASCII representation...")
    try:
        ascii_graph = graph.draw_ascii()
        print(ascii_graph)
    except:
        print("Could not generate visualization")

print("\n" + "=" * 70)
