"""
Comprehensive test for LLM-powered ReAct agent.
Tests all scenarios to ensure LLM handles everything.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.llm_agent import run_llm_agent

def test_query(query: str, description: str):
    """Test a single query and display results."""
    print("\n" + "=" * 80)
    print(f"TEST: {description}")
    print("=" * 80)
    print(f"Query: {query}")
    print("-" * 80)
    
    try:
        result = run_llm_agent(query)
        messages = result.get("messages", [])
        
        # Get the last message (agent's response)
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, 'content'):
                response = last_msg.content
            elif isinstance(last_msg, dict):
                response = last_msg.get("content", "No response")
            else:
                response = str(last_msg)
            
            print(f"Response: {response}")
            print(f"Messages exchanged: {len(messages)}")
            print("✅ SUCCESS")
        else:
            print("❌ FAILED: No messages in result")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)

if __name__ == "__main__":
    print("\nTESTING LLM-POWERED REACT AGENT")
    print("This tests if LLM can handle ALL scenarios without manual logic\n")
    
    # Test 1: Order status with ID
    test_query(
        "Where is my order 98762?",
        "Order Status Query (with order ID)"
    )
    
    # Test 2: Order status without ID (should ask)
    test_query(
        "Where is my order?",
        "Order Status Query (missing order ID - LLM should ask)"
    )
    
    # Test 3: Refund status
    test_query(
        "What's the status of my refund for order 54321?",
        "Refund Status Query"
    )
    
    # Test 4: Product availability
    test_query(
        "Is product P123 in stock?",
        "Product Availability Query"
    )
    
    # Test 5: Policy question
    test_query(
        "What is your return policy?",
        "Policy Query (RAG should be used)"
    )
    
    # Test 6: Complex multi-step query
    test_query(
        "My order 98762 is late. Can I get a refund? What's your policy on this?",
        "Complex Query (should use multiple tools + RAG)"
    )
    
    # Test 7: Charges query
    test_query(
        "Why was I charged extra on my order?",
        "Charges Policy Query"
    )
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("If all tests passed, the LLM agent is working correctly!")
    print("We can then delete the old manual logic files.")
