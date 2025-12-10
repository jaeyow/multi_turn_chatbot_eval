"""Quick test of the refactored decide_mode function using Instructor."""
import asyncio
from application import decide_mode, State


async def test_decide_mode():
    """Test the instructor-based mode classification."""
    
    test_queries = [
        ("What are your opening hours?", "shop_info"),
        ("Do you have mountain bikes?", "product_inquiry"),
        ("I need to book a repair appointment", "book_appointment"),
        ("How do I clean my bike chain?", "maintenance_tips"),
        ("What's your return policy?", "policy_question"),
        ("What can you help me with?", "what_can_you_do"),
        ("What was my appointment again?", "recall_booking"),
        ("Tell me a joke", "unknown"),
    ]
    
    print("Testing Instructor-based mode classification:\n")
    print("=" * 80)
    
    for query, expected_mode in test_queries:
        state = State({"query": query})
        result, updated_state = await decide_mode(state)
        
        actual_mode = result["mode"]
        status = "✓" if actual_mode == expected_mode else "✗"
        
        print(f"{status} Query: {query}")
        print(f"  Expected: {expected_mode}")
        print(f"  Got: {actual_mode}")
        print()


if __name__ == "__main__":
    asyncio.run(test_decide_mode())
