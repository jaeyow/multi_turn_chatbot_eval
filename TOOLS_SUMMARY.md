# Conversation Extraction Tools - Summary

## 🎯 Purpose

Extract and prepare conversation data from Burr LocalTrackingClient storage files for multi-turn evaluation and analysis.

## 🛠️ Tools Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Burr Tracking Storage                    │
│                        (~/.burr/)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Project 1  │  │   Project 2  │  │   Project N  │    │
│  │              │  │              │  │              │    │
│  │ • log.jsonl  │  │ • log.jsonl  │  │ • log.jsonl  │    │
│  │ • metadata   │  │ • metadata   │  │ • metadata   │    │
│  │ • graph      │  │ • graph      │  │ • graph      │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Extraction & Processing Layer                  │
│                                                             │
│  extract_conversations.py   conversation_extraction.ipynb  │
│  • Command-line tool        • Interactive notebook         │
│  • Batch processing         • Visual analysis              │
│  • Multiple formats         • Exploration                  │
│                                                             │
│  evaluation_helpers.py      example_extraction.py          │
│  • Filter & transform       • Complete workflow            │
│  • Format conversion        • Usage examples               │
│  • Metrics calculation      • Best practices               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Output Data Formats                      │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Raw JSON   │  │  LangChain   │  │    Ragas     │    │
│  │  (Complete)  │  │   Format     │  │   Format     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   CSV Data   │  │  Annotation  │  │   Summary    │    │
│  │  (Tabular)   │  │   Template   │  │   Metrics    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Evaluation & Analysis                          │
│                                                             │
│  • LangChain Evaluation                                     │
│  • Ragas Evaluation                                         │
│  • Custom Metrics                                           │
│  • Human Annotation                                         │
│  • Statistical Analysis                                     │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Data Extracted

### Conversation Structure
```
Conversation
├── project_id: UUID
├── app_id: UUID
├── total_turns: int
├── turns: []
│   └── Turn
│       ├── turn_number: int
│       └── messages: []
│           └── Message
│               ├── role: "user" | "assistant"
│               ├── content: str
│               └── type: "text" | "image" | ...
├── actions: []
│   └── Action
│       ├── action: str (name)
│       ├── sequence_id: int
│       ├── start_time: timestamp
│       ├── end_time: timestamp
│       ├── inputs: dict
│       ├── result: dict
│       └── state: dict (full state snapshot)
├── state_history: []
└── metadata: {}
```

## 🔄 Workflow

### Simple Workflow
```bash
# 1. Extract all conversations
python extract_conversations.py --output data.json --summary

# 2. View summary
cat data_summary.json

# 3. Prepare for evaluation
python evaluation_helpers.py data.json
```

### Advanced Workflow
```python
# 1. Extract
from extract_conversations import BurrConversationExtractor
extractor = BurrConversationExtractor()
conversations = extractor.extract_all_conversations()

# 2. Filter
from evaluation_helpers import ConversationEvaluationHelper
helper = ConversationEvaluationHelper()
multi_turn = helper.filter_by_turn_count(conversations, min_turns=2)

# 3. Format
langchain_data = helper.format_for_langchain(multi_turn)

# 4. Evaluate
# Use with your evaluation framework
```

## 📈 Analysis Capabilities

### Metrics Calculated
- **Turn Distribution**: How many conversations have 1, 2, 3+ turns
- **Message Length**: Average user/assistant message lengths
- **Action Patterns**: Common action execution sequences
- **Conversation Duration**: Time taken per conversation
- **State Evolution**: How state changes across turns

### Visualizations (in Notebook)
- Turn distribution histograms
- Message length distributions
- Turn progression analysis
- Conversation flow patterns
- Action timing analysis

## 🎨 Output Formats

### 1. Raw Extraction Format
Complete conversation data with all details
```json
{
  "project_id": "...",
  "app_id": "...",
  "turns": [...],
  "actions": [...],
  "state_history": [...]
}
```

### 2. LangChain Format
Ready for LangChain evaluation
```json
{
  "input": "user query",
  "expected_output": "assistant response",
  "context": ["previous messages"],
  "conversation_id": "...",
  "turn_number": 1
}
```

### 3. Ragas Format
Ready for Ragas evaluation
```json
{
  "question": "user query",
  "answer": "assistant response",
  "contexts": ["context1", "context2"],
  "conversation_id": "...",
  "turn_number": 1
}
```

### 4. CSV Format
Tabular data for analysis in Excel/Pandas
```csv
conversation_id,turn_number,user_message,assistant_message,user_length,assistant_length
...
```

### 5. Annotation Template
For human evaluation
```json
{
  "conversation_id": "...",
  "turn_number": 1,
  "user_input": "...",
  "assistant_response": "...",
  "annotations": {
    "relevance": null,
    "accuracy": null,
    "helpfulness": null,
    "coherence": null
  },
  "notes": ""
}
```

## 🎯 Use Cases

### 1. Single-Turn Evaluation
Extract and evaluate simple Q&A interactions
```python
single_turn = helper.filter_by_turn_count(conversations, max_turns=1)
```

### 2. Multi-Turn Context Evaluation
Evaluate context retention across turns
```python
multi_turn = helper.filter_by_turn_count(conversations, min_turns=2)
```

### 3. Action Pattern Analysis
Identify common conversation flows
```python
sequences = helper.extract_action_sequences(conversations)
```

### 4. Quality Assurance
Prepare data for human annotation
```python
helper.export_for_human_annotation(conversations, "annotation.json")
```

### 5. Performance Analysis
Analyze response times and efficiency
```python
# Extract timing data from actions
for action in conversation['actions']:
    duration = action['end_time'] - action['start_time']
    # Analyze...
```

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `QUICK_START.md` | Quick reference and commands |
| `CONVERSATION_EXTRACTION_GUIDE.md` | Complete usage guide |
| `TOOLS_SUMMARY.md` | This file - overview |
| `README.md` | Main project documentation |

## 💻 Example Code Snippets

### Extract from specific project
```python
extractor = BurrConversationExtractor()
conversations = extractor.extract_all_conversations(project_id="your-project-id")
extractor.save_to_json(conversations, "project_conversations.json")
```

### Filter and format
```python
helper = ConversationEvaluationHelper()

# Get multi-turn conversations
multi = helper.filter_by_turn_count(conversations, min_turns=2)

# Format for LangChain
langchain_data = helper.format_for_langchain(multi)

# Save
import json
with open('eval_data.json', 'w') as f:
    json.dump(langchain_data, f, indent=2)
```

### Calculate metrics
```python
metrics = helper.calculate_conversation_metrics(conversations)
print(f"Total conversations: {metrics['total_conversations']}")
print(f"Average turns: {metrics['avg_turns_per_conversation']:.2f}")
print(f"Message stats: {metrics['message_stats']}")
```

### Group by length
```python
groups = helper.group_by_conversation_length(conversations)
print(f"Single-turn: {len(groups['single_turn'])}")
print(f"Short (2-3): {len(groups['short'])}")
print(f"Medium (4-6): {len(groups['medium'])}")
print(f"Long (7+): {len(groups['long'])}")
```

## ✅ Next Steps

1. ✅ Extract your conversations
2. ✅ Analyze and filter the data
3. ✅ Choose evaluation framework
4. ✅ Format data appropriately
5. ✅ Define evaluation metrics
6. ✅ Run evaluation
7. ✅ Iterate and improve

## 🔗 Integration Examples

### With LangChain
```python
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("qa")
langchain_data = helper.format_for_langchain(conversations)

for item in langchain_data:
    result = evaluator.evaluate_strings(
        prediction=item['expected_output'],
        input=item['input']
    )
```

### With Ragas
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

ragas_data = helper.format_for_ragas(conversations)
result = evaluate(ragas_data, metrics=[faithfulness, answer_relevancy])
```

### With Custom Metrics
```python
def custom_metric(conversation):
    score = 0
    for turn in conversation['turns']:
        # Your logic here
        score += calculate_turn_score(turn)
    return score / len(conversation['turns'])

scores = [custom_metric(c) for c in conversations]
avg_score = sum(scores) / len(scores)
```

## 🎓 Best Practices

1. **Review Sample Data**: Always check a few conversations manually first
2. **Filter Appropriately**: Use turn count filters to focus on relevant data
3. **Preserve Context**: Include conversation history for multi-turn evaluation
4. **Calculate Baselines**: Establish baseline metrics before improvements
5. **Version Your Data**: Keep track of extraction dates and configurations
6. **Document Metrics**: Clearly define what you're measuring
7. **Iterative Analysis**: Extract → Analyze → Improve → Repeat
