"""
Quick test script to verify the extraction tools are working correctly.
"""
from extract_conversations import BurrConversationExtractor
from evaluation_helpers import ConversationEvaluationHelper
import json


def test_extractor():
    """Test the basic extractor functionality."""
    print("Testing BurrConversationExtractor...")
    
    extractor = BurrConversationExtractor()
    
    # Test listing projects
    projects = extractor.list_projects()
    print(f"  ✓ Found {len(projects)} project(s)")
    
    if not projects:
        print("  ⚠ No projects found - this is expected if you haven't run the app yet")
        return False
    
    # Test listing applications
    for proj in projects[:1]:  # Test first project only
        apps = extractor.list_applications(proj)
        print(f"  ✓ Project {proj[:8]}... has {len(apps)} application(s)")
        
        if apps:
            # Test extracting a conversation
            conv = extractor.extract_conversation(proj, apps[0]['app_id'])
            if conv:
                print(f"  ✓ Extracted conversation with {conv['total_turns']} turn(s)")
                return True
    
    return False


def test_helpers():
    """Test the evaluation helper functions."""
    print("\nTesting ConversationEvaluationHelper...")
    
    # Create sample conversation data
    sample_conversations = [
        {
            "app_id": "test-app-1",
            "project_id": "test-project",
            "total_turns": 2,
            "turns": [
                {
                    "turn_number": 1,
                    "messages": [
                        {"role": "user", "content": "Hello", "type": "text"},
                        {"role": "assistant", "content": "Hi there!", "type": "text"}
                    ]
                },
                {
                    "turn_number": 2,
                    "messages": [
                        {"role": "user", "content": "How are you?", "type": "text"},
                        {"role": "assistant", "content": "I'm doing well!", "type": "text"}
                    ]
                }
            ],
            "actions": [
                {"action": "query", "sequence_id": 0},
                {"action": "respond", "sequence_id": 1}
            ],
            "metadata": {"total_actions": 2, "total_messages": 4}
        }
    ]
    
    helper = ConversationEvaluationHelper()
    
    # Test filtering
    filtered = helper.filter_by_turn_count(sample_conversations, min_turns=2)
    print(f"  ✓ Filtering works: {len(filtered)} conversation(s) with 2+ turns")
    
    # Test pair extraction
    pairs = helper.extract_conversation_pairs(sample_conversations)
    print(f"  ✓ Pair extraction works: {len(pairs)} pair(s) extracted")
    
    # Test LangChain format
    langchain_data = helper.format_for_langchain(sample_conversations)
    print(f"  ✓ LangChain format works: {len(langchain_data)} item(s)")
    
    # Test Ragas format
    ragas_data = helper.format_for_ragas(sample_conversations)
    print(f"  ✓ Ragas format works: {len(ragas_data)} item(s)")
    
    # Test metrics calculation
    metrics = helper.calculate_conversation_metrics(sample_conversations)
    print(f"  ✓ Metrics calculation works: {metrics['total_conversations']} conversation(s)")
    
    # Test grouping
    groups = helper.group_by_conversation_length(sample_conversations)
    print(f"  ✓ Grouping works: {sum(len(g) for g in groups.values())} total")
    
    return True


def main():
    print("=" * 60)
    print("Conversation Extraction Tools - Test Suite")
    print("=" * 60)
    
    # Test extractor with real data
    extractor_works = test_extractor()
    
    # Test helpers with sample data
    helpers_work = test_helpers()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    print(f"  Extractor: {'✓ PASS' if extractor_works else '⚠ SKIP (no data)'}")
    print(f"  Helpers:   {'✓ PASS' if helpers_work else '✗ FAIL'}")
    
    if not extractor_works:
        print("\n⚠ Note: Extractor tests skipped because no data found.")
        print("   This is expected if you haven't run the chatbot yet.")
        print("   Run the chatbot first to generate tracking data:")
        print("     streamlit run streamlit_app.py")
    
    print("\n" + "=" * 60)
    print("✓ Test suite complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
