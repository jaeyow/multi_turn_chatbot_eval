# Burr Conversation Extraction Guide

This guide explains how to extract conversation data from Burr LocalTrackingClient storage for multi-turn evaluation.

## Overview

The Burr tracking system stores conversation data in `~/.burr/` with the following structure:
```
~/.burr/
├── {project_id}/
│   ├── {app_id}/
│   │   ├── log.jsonl          # Sequential log of all actions and state changes
│   │   ├── metadata.json       # Application metadata
│   │   └── graph.json          # State machine graph definition
```

## Tools Provided

### 1. `extract_conversations.py` - Command-Line Extraction Tool

Extract conversation data from Burr tracking files into structured JSON format.

#### Features:
- Extract all conversations from all projects or a specific project
- Generate structured JSON with turns, messages, and metadata
- Create summary statistics
- Support for both single-turn and multi-turn conversations

#### Usage:

**List all available projects:**
```bash
python extract_conversations.py --list-projects
```

**Extract all conversations:**
```bash
python extract_conversations.py --output my_conversations.json
```

**Extract from a specific project:**
```bash
python extract_conversations.py --project-id YOUR_PROJECT_ID --output project_conversations.json
```

**Extract a specific conversation (requires both project-id and app-id):**
```bash
python extract_conversations.py --project-id YOUR_PROJECT_ID --app-id YOUR_APP_ID --output single_conversation.json
```

**Extract with summary:**
```bash
python extract_conversations.py --output conversations.json --summary
```

This creates:
- `conversations.json` - Full conversation data
- `conversations_summary.json` - Statistical summary

#### Command-Line Options:
- `--storage-dir`: Path to Burr storage (default: `~/.burr`)
- `--project-id`: Specific project to extract (default: all)
- `--app-id`: Specific application ID to extract (requires `--project-id`)
- `--output`: Output file path (default: `extracted_conversations.json`)
- `--summary`: Also generate a summary file
- `--list-projects`: List available projects and exit

### 2. `conversation_extraction.ipynb` - Interactive Analysis Notebook

Jupyter notebook for interactive exploration and analysis of extracted conversations.

#### Features:
- Extract conversations interactively
- Visualize conversation patterns
- Analyze turn distributions and message lengths
- Export data in multiple formats (JSON, CSV)
- Generate multi-turn evaluation metrics

#### Running the Notebook:
```bash
jupyter notebook conversation_extraction.ipynb
```

Or in VS Code, just open the notebook file.

## Output Format

### Extracted Conversation Structure

```json
{
  "project_id": "your-project-id",
  "app_id": "application-uuid",
  "total_turns": 3,
  "turns": [
    {
      "turn_number": 1,
      "messages": [
        {
          "role": "user",
          "content": "User's question or input",
          "type": "text",
          "message_index": 0
        },
        {
          "role": "assistant",
          "content": "Assistant's response",
          "type": "text",
          "message_index": 1
        }
      ]
    }
  ],
  "actions": [
    {
      "action": "process_query",
      "start_time": "2025-08-01T00:05:48.898090",
      "end_time": "2025-08-01T00:05:48.900202",
      "sequence_id": 0,
      "inputs": {...},
      "result": {...},
      "state": {...}
    }
  ],
  "state_history": [...],
  "metadata": {
    "total_actions": 15,
    "total_messages": 6,
    "extracted_at": "2025-12-17T10:30:00"
  }
}
```

### Evaluation Format

The notebook also exports data in an evaluation-friendly format:

```json
{
  "conversation_id": "app-uuid",
  "project_id": "project-id",
  "turns": [
    {
      "turn_number": 1,
      "user_input": {
        "text": "User message",
        "type": "text"
      },
      "assistant_response": {
        "text": "Assistant response",
        "type": "text"
      },
      "context": []
    }
  ],
  "metadata": {...}
}
```

## Data Fields Explained

### Turn-Level Data:
- **turn_number**: Sequential number of the turn (1-indexed)
- **messages**: Array of messages in this turn (typically user → assistant)
- **user_input**: The user's query or input
- **assistant_response**: The assistant's response

### Message Fields:
- **role**: Either "user" or "assistant"
- **content**: The actual text content
- **type**: Message type (usually "text")
- **message_index**: Position in the full conversation history

### Action Data:
- **action**: Name of the Burr action executed
- **start_time/end_time**: Timestamps for the action
- **sequence_id**: Order of execution
- **inputs**: Input parameters to the action
- **result**: Output from the action
- **state**: Full application state after the action
- **streaming**: Whether this action was streamed
- **items_streamed**: Number of streamed items (if applicable)

### Metadata:
- **total_actions**: Number of Burr actions executed
- **total_messages**: Total messages in conversation
- **extracted_at**: When the data was extracted

## Use Cases

### 1. Single-Turn Evaluation
Extract conversations with 1 turn for simple Q&A evaluation:
```python
from extract_conversations import BurrConversationExtractor

extractor = BurrConversationExtractor()
conversations = extractor.extract_all_conversations()

single_turn = [c for c in conversations if c['total_turns'] == 1]
```

### 2. Multi-Turn Evaluation
Extract multi-turn conversations to evaluate context retention:
```python
multi_turn = [c for c in conversations if c['total_turns'] > 1]
```

### 3. Action Analysis
Analyze which actions were executed and in what order:
```python
for conv in conversations:
    action_sequence = [a['action'] for a in conv['actions']]
    print(f"Conversation {conv['app_id']}: {' → '.join(action_sequence)}")
```

### 4. State Analysis
Examine how state evolved throughout the conversation:
```python
for state_snapshot in conv['state_history']:
    print(f"After {state_snapshot['action']}:")
    print(f"  - Chat history length: {len(state_snapshot['state'].get('chat_history', []))}")
    print(f"  - In appointment flow: {state_snapshot['state'].get('in_appointment_flow', False)}")
```

## Integration with Evaluation Frameworks

### Example: Prepare data for LangChain evaluation

```python
from extract_conversations import BurrConversationExtractor

extractor = BurrConversationExtractor()
conversations = extractor.extract_all_conversations()

# Convert to LangChain format
eval_dataset = []
for conv in conversations:
    for turn in conv['turns']:
        user_msg = next((m for m in turn['messages'] if m['role'] == 'user'), None)
        asst_msg = next((m for m in turn['messages'] if m['role'] == 'assistant'), None)
        
        if user_msg and asst_msg:
            eval_dataset.append({
                'input': user_msg['content'],
                'expected_output': asst_msg['content'],
                'conversation_id': conv['app_id'],
                'turn_number': turn['turn_number']
            })
```

### Example: Export for Ragas evaluation

```python
import pandas as pd

# Create DataFrame for Ragas
rows = []
for conv in conversations:
    for turn in conv['turns']:
        user_msg = next((m for m in turn['messages'] if m['role'] == 'user'), None)
        asst_msg = next((m for m in turn['messages'] if m['role'] == 'assistant'), None)
        
        if user_msg and asst_msg:
            rows.append({
                'question': user_msg['content'],
                'answer': asst_msg['content'],
                'contexts': [],  # Add context if needed
                'ground_truth': ''  # Add ground truth if available
            })

df = pd.DataFrame(rows)
df.to_csv('ragas_eval_data.csv', index=False)
```

## Metrics for Multi-Turn Evaluation

The extraction tools provide several metrics useful for evaluation:

1. **Turn Count Distribution**: How many conversations have 1, 2, 3+ turns
2. **Message Length Statistics**: Average length of user/assistant messages
3. **Action Execution Patterns**: Which actions are most common
4. **State Persistence**: How state evolves across turns
5. **Timing Information**: Duration of each action

## Tips for Effective Extraction

1. **Filter by Project**: If you have multiple projects, extract them separately
2. **Check Data Quality**: Review a few conversations manually before bulk processing
3. **Handle Incomplete Conversations**: Some conversations may not have completed - filter these out if needed
4. **Preserve Context**: The `state_history` field contains full state - use this for context-aware evaluation
5. **Action Sequences**: Analyze action patterns to understand conversation flow

## Troubleshooting

**No conversations found:**
- Check that `~/.burr` exists and contains data
- Verify you're using the correct storage directory
- Ensure applications have been run with tracking enabled

**Empty turns:**
- Some applications may not populate `chat_history` in state
- Check the `state_history` field for alternative data sources

**Missing fields:**
- Not all applications track the same state fields
- Adjust extraction logic based on your application's state structure

## Next Steps

After extracting conversations:

1. **Manual Review**: Inspect a sample of conversations for quality
2. **Ground Truth Creation**: Create expected responses for evaluation
3. **Metric Selection**: Choose appropriate metrics (accuracy, relevance, coherence, etc.)
4. **Baseline Establishment**: Run initial evaluation to establish baseline
5. **Iterative Improvement**: Use insights to improve the chatbot

## Additional Resources

- [Burr Documentation](https://burr.dagworks.io/)
- [Burr Tracking Guide](https://burr.dagworks.io/concepts/tracking/)
- [Multi-Turn Evaluation Best Practices](https://python.langchain.com/docs/guides/evaluation)
