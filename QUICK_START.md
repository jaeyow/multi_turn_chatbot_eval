# Quick Start: Conversation Extraction

## 🚀 Quick Commands

### List available projects
```bash
python extract_conversations.py --list-projects
```

### Extract all conversations
```bash
python extract_conversations.py --output my_conversations.json --summary
```

### Extract specific project
```bash
python extract_conversations.py --project-id YOUR_PROJECT_ID --output project_convs.json
```

### Run complete example
```bash
python example_extraction.py
```

### Open interactive notebook
```bash
jupyter notebook conversation_extraction.ipynb
```

## 📁 Files Created

| File | Description |
|------|-------------|
| **Core Tools** | |
| `extract_conversations.py` | Command-line extraction tool |
| `conversation_extraction.ipynb` | Interactive analysis notebook |
| `evaluation_helpers.py` | Helper functions for evaluation |
| `example_extraction.py` | Complete usage example |
| **Generated Data** | |
| `extracted_conversations.json` | Full conversation data |
| `extraction_summary.json` | Statistical summary |
| `langchain_eval_data.json` | LangChain format |
| `ragas_eval_data.json` | Ragas format |
| `human_annotation_template.json` | Annotation template |
| `conversation_turns.csv` | Flattened CSV data |
| `conversation_metrics.csv` | Metrics CSV |
| **Documentation** | |
| `CONVERSATION_EXTRACTION_GUIDE.md` | Complete guide |
| `QUICK_START.md` | This file |

## 💡 Common Use Cases

### Use Case 1: Extract and Analyze All Conversations
```bash
# Extract
python extract_conversations.py --output all_convs.json --summary

# Analyze
python evaluation_helpers.py all_convs.json
```

### Use Case 2: Filter Multi-Turn Conversations
```python
from extract_conversations import BurrConversationExtractor
from evaluation_helpers import ConversationEvaluationHelper

extractor = BurrConversationExtractor()
conversations = extractor.extract_all_conversations()

helper = ConversationEvaluationHelper()
multi_turn = helper.filter_by_turn_count(conversations, min_turns=2)
print(f"Found {len(multi_turn)} multi-turn conversations")
```

### Use Case 3: Prepare for LangChain Evaluation
```python
from evaluation_helpers import ConversationEvaluationHelper
import json

with open('extracted_conversations.json', 'r') as f:
    conversations = json.load(f)

helper = ConversationEvaluationHelper()
langchain_data = helper.format_for_langchain(conversations)

with open('langchain_eval.json', 'w') as f:
    json.dump(langchain_data, f, indent=2)
```

### Use Case 4: Interactive Analysis
```bash
# Open notebook
jupyter notebook conversation_extraction.ipynb

# Run all cells to:
# - Extract conversations
# - Generate visualizations
# - Calculate metrics
# - Export in multiple formats
```

## 📊 What Data is Extracted?

### Turn-Level Data
- User messages
- Assistant responses
- Turn numbers
- Message types

### Action-Level Data
- Action names
- Execution order (sequence_id)
- Start/end times
- Inputs and results
- State snapshots

### Conversation-Level Data
- Total turns
- Total actions
- Full state history
- Metadata (timestamps, etc.)

## 🔍 Key Functions

### `BurrConversationExtractor`
- `list_projects()` - List all projects
- `list_applications(project_id)` - List apps in a project
- `extract_conversation(project_id, app_id)` - Extract single conversation
- `extract_all_conversations()` - Extract all conversations
- `save_to_json()` - Save to JSON file

### `ConversationEvaluationHelper`
- `filter_by_turn_count()` - Filter by number of turns
- `extract_conversation_pairs()` - Get user-assistant pairs
- `format_for_langchain()` - LangChain format
- `format_for_ragas()` - Ragas format
- `calculate_conversation_metrics()` - Calculate metrics
- `export_for_human_annotation()` - Annotation template

## 🎯 Typical Workflow

1. **Extract** conversations from Burr tracking
   ```bash
   python extract_conversations.py --output data.json --summary
   ```

2. **Analyze** the extracted data
   ```bash
   python evaluation_helpers.py data.json
   ```

3. **Filter** for your use case
   ```python
   # Single-turn only
   single = helper.filter_by_turn_count(conversations, max_turns=1)
   
   # Multi-turn only
   multi = helper.filter_by_turn_count(conversations, min_turns=2)
   ```

4. **Format** for evaluation framework
   ```python
   langchain_data = helper.format_for_langchain(conversations)
   # or
   ragas_data = helper.format_for_ragas(conversations)
   ```

5. **Evaluate** using your chosen framework
   ```python
   # Use with your evaluation framework
   # LangChain, Ragas, custom metrics, etc.
   ```

## 📝 Example Output Structure

```json
{
  "project_id": "73513cab-1ab1-434d-b503-17b13619c946",
  "app_id": "5dc14f75-b580-4b61-a5f5-228133d7faf6",
  "total_turns": 2,
  "turns": [
    {
      "turn_number": 1,
      "messages": [
        {
          "role": "user",
          "content": "What are your hours?",
          "type": "text"
        },
        {
          "role": "assistant",
          "content": "We're open Monday-Friday 9AM-6PM...",
          "type": "text"
        }
      ]
    }
  ],
  "actions": [...],
  "metadata": {...}
}
```

## 🆘 Troubleshooting

**No conversations found?**
- Check `~/.burr` exists
- Verify tracking was enabled in your Burr app
- Confirm you've run the application at least once

**Empty turns?**
- Check that `chat_history` is being populated in state
- Review `state_history` field for alternative data

**Missing data?**
- Not all state fields are always present
- Customize extraction logic for your app's state structure

## 📚 More Info

- Full guide: [CONVERSATION_EXTRACTION_GUIDE.md](CONVERSATION_EXTRACTION_GUIDE.md)
- Burr docs: [burr.dagworks.io](https://burr.dagworks.io/)
- Example app: [README.md](README.md)
