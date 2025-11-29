# src/main.py
"""
CLI entrypoint to run LLM-powered ReAct agent.

Usage:
    python src/main.py "Where is my order 98762?"
If no argument is provided, the script will prompt interactively.
"""
import sys
from src.agent.llm_agent import run_llm_agent_simple

def main():
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("User query: ")
    
    print("\n" + "="*70)
    print("LLM-POWERED REACT AGENT")
    print("="*70)
    print(f"Query: {q}\n")
    
    response = run_llm_agent_simple(q)
    
    print("="*70)
    print("RESPONSE:")
    print("="*70)
    print(response)
    print()

if __name__ == "__main__":
    main()
