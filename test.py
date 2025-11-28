# test_agent.py
"""
Temporary end-to-end test for the agent pipeline (ingest, retriever, tools, agent nodes).
Run from project root:
    python test_agent.py

This script is safe to remove later.
"""

from pprint import pprint
from src.agent.agent_graph import run_agent

TEST_QUERIES = [
    {
        "q": "My order 98762 says 'Out for delivery' for 3 days. What's happening?",
        "expect_intent": "order_status",
        "expect_doc": None,  # the agent should call tools primarily
    },
    {
        "q": "Is product P123 available in stock?",
        "expect_intent": "product_availability",
        "expect_doc": None,
    },
    {
        "q": "I want to check refund status for order 54321.",
        "expect_intent": "refund_status",
        "expect_doc": None,
    },
    {
        "q": "How long do refunds take according to policy?",
        "expect_intent": "return_policy",  # rule-based classifier maps refund/return questions here
        "expect_doc": "refund_policy.txt",
    },
    {
        "q": "Why was I charged extra on my last purchase?",
        "expect_intent": "charges_query",
        "expect_doc": "charges_policy.txt",
    },
    {
        "q": "Where is my order?",
        "expect_intent": "order_status",
        "expect_doc": None,
        "expect_missing_slot": "order_id",
    }
]

def brief_tool_resp(tool_resp):
    if not tool_resp:
        return None
    if isinstance(tool_resp, dict):
        # show short summary keys
        keys = list(tool_resp.keys())
        summary = {k: tool_resp.get(k) for k in ("order_id", "order_status", "refund_status", "in_stock", "product_id") if k in keys}
        # fallback: return a small excerpt
        if not summary:
            # return a couple of top-level keys with small values
            return {k: (tool_resp[k] if len(str(tool_resp[k])) < 80 else str(tool_resp[k])[:80]+"...") for k in list(tool_resp)[:4]}
        return summary
    return str(tool_resp)


def run_test_case(tc):
    print("\n" + "="*80)
    print("QUERY:", tc["q"])
    state = run_agent(tc["q"])
    print("\n--- Agent state summary ---")
    print("Intent:", state.intent)
    print("Slots:", state.slots)
    print("Tool response (brief):")
    pprint(brief_tool_resp(state.tool_response))
    print("\nRAG snippets (top):")
    if state.rag_results:
        for i, s in enumerate(state.rag_results, 1):
            meta = s.get("metadata", {})
            print(f"  {i}. doc_id={meta.get('doc_id')} combined={s.get('combined'):.3f}")
            print("     preview:", s.get("text", "")[:220].replace("\n", " ") + "...")
    else:
        print("  (none)")

    print("\nFinal answer:")
    print(state.final_answer)
    print("\nErrors:", state.errors)
    print("="*80)

    # Basic assertions (non-fatal; print warnings)
    if tc.get("expect_intent") and state.intent != tc["expect_intent"]:
        print(f"[WARN] Expected intent {tc['expect_intent']!r} but got {state.intent!r}")

    if tc.get("expect_missing_slot"):
        if tc["expect_missing_slot"] in state.slots:
            print(f"[WARN] Expected slot {tc['expect_missing_slot']} to be missing but it was present: {state.slots.get(tc['expect_missing_slot'])}")
        else:
            print(f"[OK] Slot {tc['expect_missing_slot']} is correctly missing (agent should ask).")

    if tc.get("expect_doc"):
        # check if any returned rag doc_id matches expected
        docs = [r.get("metadata", {}).get("doc_id") for r in (state.rag_results or [])]
        if tc["expect_doc"] not in docs:
            print(f"[WARN] Expected RAG doc {tc['expect_doc']} in top results but found {docs}")
        else:
            print(f"[OK] Expected policy doc {tc['expect_doc']} found in RAG results.")

def main():
    print("Starting agent end-to-end tests (temporary).")
    for tc in TEST_QUERIES:
        run_test_case(tc)
    print("\nAll tests executed. Remove test_agent.py when done.")

if __name__ == "__main__":
    main()
