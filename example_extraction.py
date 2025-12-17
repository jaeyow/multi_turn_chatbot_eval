"""
Example script demonstrating how to extract and prepare conversation data
from Burr tracking files for multi-turn evaluation.
"""
import json
from extract_conversations import BurrConversationExtractor
from evaluation_helpers import (
    ConversationEvaluationHelper,
    print_conversation_summary
)


def main():
    print("=" * 70)
    print("Burr Conversation Extraction Example")
    print("=" * 70)
    
    # Step 1: Initialize the extractor
    print("\n1. Initializing extractor...")
    extractor = BurrConversationExtractor(burr_storage_dir="~/.burr")
    
    # Step 2: List available projects
    print("\n2. Listing available projects...")
    projects = extractor.list_projects()
    print(f"   Found {len(projects)} project(s)")
    
    if not projects:
        print("   No projects found in ~/.burr")
        print("   Make sure you have run your Burr application with tracking enabled")
        return
    
    for proj in projects:
        apps = extractor.list_applications(proj)
        print(f"   - Project: {proj}")
        print(f"     Applications: {len(apps)}")
    
    # Step 3: Extract all conversations
    print("\n3. Extracting conversations...")
    conversations = extractor.extract_all_conversations()
    
    if not conversations:
        print("   No conversations found!")
        return
    
    print(f"   ✓ Extracted {len(conversations)} conversation(s)")
    
    # Step 4: Save raw extraction
    print("\n4. Saving extracted conversations...")
    extractor.save_to_json(conversations, "extracted_conversations.json")
    
    # Step 5: Generate summary
    print("\n5. Generating summary...")
    summary = extractor.generate_summary(conversations)
    with open("extraction_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print("   ✓ Saved summary to extraction_summary.json")
    
    # Print summary to console
    print_conversation_summary(conversations)
    
    # Step 6: Prepare data for evaluation
    print("\n6. Preparing data for evaluation...")
    helper = ConversationEvaluationHelper()
    
    # Filter conversations
    multi_turn = helper.filter_by_turn_count(conversations, min_turns=2)
    print(f"   - Multi-turn conversations: {len(multi_turn)}")
    
    single_turn = helper.filter_by_turn_count(conversations, max_turns=1)
    print(f"   - Single-turn conversations: {len(single_turn)}")
    
    # Extract pairs
    pairs = helper.extract_conversation_pairs(conversations)
    print(f"   - Total user-assistant pairs: {len(pairs)}")
    
    # Step 7: Export in different formats
    print("\n7. Exporting in evaluation formats...")
    
    # LangChain format
    langchain_data = helper.format_for_langchain(conversations)
    with open("langchain_eval_data.json", 'w') as f:
        json.dump(langchain_data, f, indent=2)
    print(f"   ✓ LangChain format: langchain_eval_data.json ({len(langchain_data)} items)")
    
    # Ragas format
    ragas_data = helper.format_for_ragas(conversations)
    with open("ragas_eval_data.json", 'w') as f:
        json.dump(ragas_data, f, indent=2)
    print(f"   ✓ Ragas format: ragas_eval_data.json ({len(ragas_data)} items)")
    
    # Human annotation format
    helper.export_for_human_annotation(
        conversations,
        "human_annotation_template.json",
        fields=['relevance', 'accuracy', 'helpfulness', 'coherence']
    )
    
    # Step 8: Group by conversation length
    print("\n8. Grouping by conversation length...")
    groups = helper.group_by_conversation_length(conversations)
    for category, convs in groups.items():
        print(f"   - {category}: {len(convs)} conversation(s)")
    
    # Step 9: Analyze action patterns
    print("\n9. Analyzing action patterns...")
    sequences = helper.extract_action_sequences(conversations)
    
    # Find unique patterns
    unique_patterns = {}
    for seq in sequences:
        pattern = seq['sequence_string']
        if pattern not in unique_patterns:
            unique_patterns[pattern] = 0
        unique_patterns[pattern] += 1
    
    print(f"   Found {len(unique_patterns)} unique action patterns")
    print("\n   Top 5 patterns:")
    for pattern, count in sorted(unique_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"     - {pattern}: {count} occurrence(s)")
    
    # Step 10: Show example conversation
    print("\n10. Example Conversation:")
    print("=" * 70)
    if conversations:
        example = conversations[0]
        conv_summary = helper.get_conversation_summary(example)
        print(json.dumps(conv_summary, indent=2))
    
    print("\n" + "=" * 70)
    print("✓ Extraction Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - extracted_conversations.json       (Full extraction)")
    print("  - extraction_summary.json            (Statistical summary)")
    print("  - langchain_eval_data.json           (LangChain format)")
    print("  - ragas_eval_data.json               (Ragas format)")
    print("  - human_annotation_template.json     (Human annotation)")
    print("\nNext steps:")
    print("  1. Review the extracted conversations")
    print("  2. Choose an evaluation framework (LangChain, Ragas, custom)")
    print("  3. Define evaluation metrics")
    print("  4. Run evaluations and iterate!")
    print("=" * 70)


if __name__ == "__main__":
    main()
