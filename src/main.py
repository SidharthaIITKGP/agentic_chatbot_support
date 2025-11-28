# src/main.py
"""
CLI entrypoint to run agent graph directly.

Usage:
    python src/main.py "Where is my order 98762?"
If no argument is provided, the script will prompt interactively.
"""
import sys
from src.agent.agent_graph import run_agent
from pprint import pprint

def main():
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("User query: ")
    state = run_agent(q)
    print("=== FINAL ANSWER ===")
    print(state.final_answer)
    print("\n--- Tool response (brief) ---")
    pprint(state.tool_response)
    print("\n--- RAG snippets (top) ---")
    for r in (state.rag_results or []):
        print(r.get("metadata", {}).get("doc_id"), r.get("combined"))
        print(r.get("text", "")[:400])
        print("------")

if __name__ == "__main__":
    main()
