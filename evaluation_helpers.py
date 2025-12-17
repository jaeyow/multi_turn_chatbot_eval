"""
Helper functions for multi-turn conversation evaluation.

This module provides utilities for preparing extracted conversation data
for various evaluation frameworks and metrics.
"""
from typing import List, Dict, Optional, Any
from collections import defaultdict
import json


class ConversationEvaluationHelper:
    """Helper class for preparing conversation data for evaluation."""
    
    @staticmethod
    def filter_by_turn_count(
        conversations: List[Dict],
        min_turns: Optional[int] = None,
        max_turns: Optional[int] = None
    ) -> List[Dict]:
        """
        Filter conversations by number of turns.
        
        Args:
            conversations: List of extracted conversations
            min_turns: Minimum number of turns (inclusive)
            max_turns: Maximum number of turns (inclusive)
            
        Returns:
            Filtered list of conversations
        """
        filtered = conversations
        
        if min_turns is not None:
            filtered = [c for c in filtered if c['total_turns'] >= min_turns]
        
        if max_turns is not None:
            filtered = [c for c in filtered if c['total_turns'] <= max_turns]
        
        return filtered
    
    @staticmethod
    def extract_conversation_pairs(conversations: List[Dict]) -> List[Dict]:
        """
        Extract user-assistant message pairs from conversations.
        
        Returns a flat list of all user-assistant exchanges.
        
        Args:
            conversations: List of extracted conversations
            
        Returns:
            List of dicts with 'user_input', 'assistant_response', metadata
        """
        pairs = []
        
        for conv in conversations:
            for turn in conv['turns']:
                user_msg = None
                asst_msg = None
                
                for msg in turn['messages']:
                    if msg['role'] == 'user':
                        user_msg = msg
                    elif msg['role'] == 'assistant':
                        asst_msg = msg
                
                if user_msg and asst_msg:
                    pairs.append({
                        'conversation_id': conv['app_id'],
                        'project_id': conv['project_id'],
                        'turn_number': turn['turn_number'],
                        'user_input': user_msg['content'],
                        'assistant_response': asst_msg['content'],
                        'user_message_type': user_msg['type'],
                        'assistant_message_type': asst_msg['type']
                    })
        
        return pairs
    
    @staticmethod
    def extract_conversation_history(conversation: Dict) -> List[Dict]:
        """
        Extract full conversation history in chronological order.
        
        Args:
            conversation: Single conversation dict
            
        Returns:
            Ordered list of all messages
        """
        history = []
        
        for turn in conversation['turns']:
            for msg in turn['messages']:
                history.append({
                    'turn_number': turn['turn_number'],
                    'role': msg['role'],
                    'content': msg['content'],
                    'type': msg['type']
                })
        
        return history
    
    @staticmethod
    def format_for_langchain(conversations: List[Dict]) -> List[Dict]:
        """
        Format conversations for LangChain evaluation.
        
        Args:
            conversations: List of extracted conversations
            
        Returns:
            List formatted for LangChain evaluation
        """
        langchain_format = []
        
        for conv in conversations:
            for turn in conv['turns']:
                user_msg = next((m for m in turn['messages'] if m['role'] == 'user'), None)
                asst_msg = next((m for m in turn['messages'] if m['role'] == 'assistant'), None)
                
                if user_msg and asst_msg:
                    langchain_format.append({
                        'input': user_msg['content'],
                        'expected_output': asst_msg['content'],
                        'conversation_id': conv['app_id'],
                        'turn_number': turn['turn_number'],
                        'context': ConversationEvaluationHelper._get_turn_context(conv, turn['turn_number'])
                    })
        
        return langchain_format
    
    @staticmethod
    def format_for_ragas(conversations: List[Dict]) -> List[Dict]:
        """
        Format conversations for Ragas evaluation framework.
        
        Args:
            conversations: List of extracted conversations
            
        Returns:
            List formatted for Ragas evaluation
        """
        ragas_format = []
        
        for conv in conversations:
            for turn in conv['turns']:
                user_msg = next((m for m in turn['messages'] if m['role'] == 'user'), None)
                asst_msg = next((m for m in turn['messages'] if m['role'] == 'assistant'), None)
                
                if user_msg and asst_msg:
                    ragas_format.append({
                        'question': user_msg['content'],
                        'answer': asst_msg['content'],
                        'contexts': ConversationEvaluationHelper._get_turn_context(conv, turn['turn_number']),
                        'conversation_id': conv['app_id'],
                        'turn_number': turn['turn_number']
                    })
        
        return ragas_format
    
    @staticmethod
    def _get_turn_context(conversation: Dict, turn_number: int) -> List[str]:
        """
        Get the conversation context up to a specific turn.
        
        Args:
            conversation: Conversation dict
            turn_number: Turn number to get context for
            
        Returns:
            List of previous messages
        """
        context = []
        
        for turn in conversation['turns']:
            if turn['turn_number'] >= turn_number:
                break
            
            for msg in turn['messages']:
                context.append(f"{msg['role']}: {msg['content']}")
        
        return context
    
    @staticmethod
    def calculate_conversation_metrics(conversations: List[Dict]) -> Dict[str, Any]:
        """
        Calculate basic metrics for a set of conversations.
        
        Args:
            conversations: List of extracted conversations
            
        Returns:
            Dict of metrics
        """
        if not conversations:
            return {}
        
        metrics = {
            'total_conversations': len(conversations),
            'total_turns': sum(c['total_turns'] for c in conversations),
            'avg_turns_per_conversation': sum(c['total_turns'] for c in conversations) / len(conversations),
            'turn_distribution': defaultdict(int),
            'action_distribution': defaultdict(int),
            'message_stats': {
                'user': {'count': 0, 'total_length': 0},
                'assistant': {'count': 0, 'total_length': 0}
            }
        }
        
        for conv in conversations:
            # Turn distribution
            metrics['turn_distribution'][conv['total_turns']] += 1
            
            # Action distribution
            for action in conv.get('actions', []):
                metrics['action_distribution'][action['action']] += 1
            
            # Message stats
            for turn in conv['turns']:
                for msg in turn['messages']:
                    role = msg['role']
                    if role in metrics['message_stats']:
                        metrics['message_stats'][role]['count'] += 1
                        metrics['message_stats'][role]['total_length'] += len(msg['content'])
        
        # Calculate averages
        for role in ['user', 'assistant']:
            count = metrics['message_stats'][role]['count']
            if count > 0:
                metrics['message_stats'][role]['avg_length'] = (
                    metrics['message_stats'][role]['total_length'] / count
                )
        
        # Convert defaultdicts to regular dicts
        metrics['turn_distribution'] = dict(metrics['turn_distribution'])
        metrics['action_distribution'] = dict(metrics['action_distribution'])
        
        return metrics
    
    @staticmethod
    def group_by_conversation_length(conversations: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group conversations by length category.
        
        Args:
            conversations: List of extracted conversations
            
        Returns:
            Dict with keys 'single_turn', 'short' (2-3), 'medium' (4-6), 'long' (7+)
        """
        groups = {
            'single_turn': [],
            'short': [],  # 2-3 turns
            'medium': [],  # 4-6 turns
            'long': []  # 7+ turns
        }
        
        for conv in conversations:
            turns = conv['total_turns']
            if turns == 1:
                groups['single_turn'].append(conv)
            elif turns <= 3:
                groups['short'].append(conv)
            elif turns <= 6:
                groups['medium'].append(conv)
            else:
                groups['long'].append(conv)
        
        return groups
    
    @staticmethod
    def extract_action_sequences(conversations: List[Dict]) -> List[Dict]:
        """
        Extract action execution sequences from conversations.
        
        Args:
            conversations: List of extracted conversations
            
        Returns:
            List of action sequence patterns
        """
        sequences = []
        
        for conv in conversations:
            if 'actions' in conv:
                action_names = [a['action'] for a in conv['actions']]
                sequences.append({
                    'conversation_id': conv['app_id'],
                    'sequence': action_names,
                    'sequence_string': ' → '.join(action_names),
                    'length': len(action_names)
                })
        
        return sequences
    
    @staticmethod
    def find_conversations_with_action(conversations: List[Dict], action_name: str) -> List[Dict]:
        """
        Find all conversations that executed a specific action.
        
        Args:
            conversations: List of extracted conversations
            action_name: Name of the action to search for
            
        Returns:
            List of conversations containing the action
        """
        return [
            conv for conv in conversations
            if any(a['action'] == action_name for a in conv.get('actions', []))
        ]
    
    @staticmethod
    def export_for_human_annotation(
        conversations: List[Dict],
        output_file: str,
        fields: Optional[List[str]] = None
    ):
        """
        Export conversations in a format suitable for human annotation.
        
        Args:
            conversations: List of extracted conversations
            output_file: Path to output JSON file
            fields: Optional list of annotation fields to include
        """
        if fields is None:
            fields = ['relevance', 'accuracy', 'helpfulness', 'coherence']
        
        annotation_data = []
        
        for conv in conversations:
            for turn in conv['turns']:
                user_msg = next((m for m in turn['messages'] if m['role'] == 'user'), None)
                asst_msg = next((m for m in turn['messages'] if m['role'] == 'assistant'), None)
                
                if user_msg and asst_msg:
                    annotation_item = {
                        'conversation_id': conv['app_id'],
                        'turn_number': turn['turn_number'],
                        'context': ConversationEvaluationHelper._get_turn_context(conv, turn['turn_number']),
                        'user_input': user_msg['content'],
                        'assistant_response': asst_msg['content'],
                        'annotations': {field: None for field in fields},
                        'notes': ''
                    }
                    annotation_data.append(annotation_item)
        
        with open(output_file, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        
        print(f"Exported {len(annotation_data)} items for annotation to {output_file}")
    
    @staticmethod
    def get_conversation_summary(conversation: Dict) -> Dict[str, Any]:
        """
        Get a summary of a single conversation.
        
        Args:
            conversation: Single conversation dict
            
        Returns:
            Summary dict
        """
        summary = {
            'conversation_id': conversation['app_id'],
            'project_id': conversation['project_id'],
            'total_turns': conversation['total_turns'],
            'total_actions': len(conversation.get('actions', [])),
            'action_sequence': [a['action'] for a in conversation.get('actions', [])],
            'turns_summary': []
        }
        
        for turn in conversation['turns']:
            turn_summary = {
                'turn_number': turn['turn_number'],
                'message_count': len(turn['messages'])
            }
            
            for msg in turn['messages']:
                key = f"{msg['role']}_preview"
                content = msg['content']
                turn_summary[key] = content[:100] + '...' if len(content) > 100 else content
            
            summary['turns_summary'].append(turn_summary)
        
        return summary


def print_conversation_summary(conversations: List[Dict]):
    """
    Print a formatted summary of conversations.
    
    Args:
        conversations: List of extracted conversations
    """
    helper = ConversationEvaluationHelper()
    metrics = helper.calculate_conversation_metrics(conversations)
    
    print("=" * 70)
    print("CONVERSATION DATASET SUMMARY")
    print("=" * 70)
    print(f"\nTotal Conversations: {metrics['total_conversations']}")
    print(f"Total Turns: {metrics['total_turns']}")
    print(f"Average Turns per Conversation: {metrics['avg_turns_per_conversation']:.2f}")
    
    print(f"\nTurn Distribution:")
    for turns, count in sorted(metrics['turn_distribution'].items()):
        print(f"  {turns} turn(s): {count} conversation(s)")
    
    print(f"\nMessage Statistics:")
    for role in ['user', 'assistant']:
        stats = metrics['message_stats'][role]
        print(f"  {role.title()}:")
        print(f"    - Count: {stats['count']}")
        print(f"    - Avg Length: {stats.get('avg_length', 0):.0f} characters")
    
    print(f"\nTop Actions:")
    sorted_actions = sorted(
        metrics['action_distribution'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for action, count in sorted_actions[:10]:
        print(f"  {action}: {count}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        
        with open(input_file, 'r') as f:
            conversations = json.load(f)
        
        print_conversation_summary(conversations)
        
        # Show sample formatting
        helper = ConversationEvaluationHelper()
        
        print("\n\nSample LangChain Format:")
        print("-" * 70)
        langchain_data = helper.format_for_langchain(conversations[:1])
        if langchain_data:
            print(json.dumps(langchain_data[0], indent=2))
        
        print("\n\nSample Ragas Format:")
        print("-" * 70)
        ragas_data = helper.format_for_ragas(conversations[:1])
        if ragas_data:
            print(json.dumps(ragas_data[0], indent=2))
    else:
        print("Usage: python evaluation_helpers.py <extracted_conversations.json>")
