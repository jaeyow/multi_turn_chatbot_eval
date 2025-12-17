"""
Extract conversation data from Burr LocalTrackingClient storage.

This script reads from the .burr tracking files and extracts conversation
data into a structured JSON format suitable for multi-turn evaluation.
"""
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse


class BurrConversationExtractor:
    """Extract and format conversation data from Burr tracking files."""
    
    def __init__(self, burr_storage_dir: str = "~/.burr"):
        self.storage_dir = Path(burr_storage_dir).expanduser()
        
    def list_projects(self) -> List[str]:
        """List all project IDs in the storage directory."""
        if not self.storage_dir.exists():
            return []
        return [d.name for d in self.storage_dir.iterdir() if d.is_dir()]
    
    def list_applications(self, project_id: str) -> List[Dict[str, str]]:
        """List all application IDs for a given project."""
        project_dir = self.storage_dir / project_id
        if not project_dir.exists():
            return []
        
        apps = []
        for app_dir in project_dir.iterdir():
            if app_dir.is_dir() and (app_dir / "log.jsonl").exists():
                metadata_file = app_dir / "metadata.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                apps.append({
                    "app_id": app_dir.name,
                    "path": str(app_dir),
                    "metadata": metadata
                })
        return apps
    
    def extract_conversation(self, project_id: str, app_id: str) -> Optional[Dict]:
        """Extract conversation data from a specific application."""
        log_file = self.storage_dir / project_id / app_id / "log.jsonl"
        
        if not log_file.exists():
            print(f"Warning: Log file not found: {log_file}")
            return None
        
        conversation = {
            "project_id": project_id,
            "app_id": app_id,
            "turns": [],
            "metadata": {},
            "state_history": [],
            "actions": []
        }
        
        # Parse the log file
        turns = []
        current_turn = {}
        action_sequence = []
        
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entry_type = entry.get("type")
                    
                    if entry_type == "begin_entry":
                        # Start of an action
                        action_sequence.append({
                            "action": entry.get("action"),
                            "start_time": entry.get("start_time"),
                            "sequence_id": entry.get("sequence_id"),
                            "inputs": entry.get("inputs", {})
                        })
                    
                    elif entry_type == "end_entry":
                        # End of an action
                        action = entry.get("action")
                        result = entry.get("result", {})
                        state = entry.get("state", {})
                        
                        # Update action sequence
                        if action_sequence and action_sequence[-1]["action"] == action:
                            action_sequence[-1].update({
                                "end_time": entry.get("end_time"),
                                "result": result,
                                "state": state,
                                "exception": entry.get("exception")
                            })
                        
                        # Extract chat history updates
                        chat_history = state.get("chat_history", [])
                        if chat_history:
                            # Store full state history
                            conversation["state_history"].append({
                                "sequence_id": entry.get("sequence_id"),
                                "action": action,
                                "state": state,
                                "timestamp": entry.get("end_time")
                            })
                    
                    elif entry_type == "begin_stream":
                        if action_sequence:
                            action_sequence[-1]["streaming"] = True
                            action_sequence[-1]["stream_start"] = entry.get("stream_init_time")
                    
                    elif entry_type == "end_stream":
                        if action_sequence:
                            action_sequence[-1]["stream_end"] = entry.get("end_time")
                            action_sequence[-1]["items_streamed"] = entry.get("items_streamed")
                
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
                    continue
        
        # Extract turns from the final state
        if conversation["state_history"]:
            final_state = conversation["state_history"][-1]["state"]
            chat_history = final_state.get("chat_history", [])
            
            # Group messages into turns (user -> assistant pairs)
            current_turn_messages = []
            turn_number = 0
            
            for i, message in enumerate(chat_history):
                role = message.get("role")
                content = message.get("content")
                msg_type = message.get("type")
                
                if role == "user":
                    # Start a new turn
                    if current_turn_messages:
                        # Save previous turn
                        turn_number += 1
                        turns.append({
                            "turn_number": turn_number,
                            "messages": current_turn_messages
                        })
                        current_turn_messages = []
                    
                    current_turn_messages.append({
                        "role": role,
                        "content": content,
                        "type": msg_type,
                        "message_index": i
                    })
                
                elif role == "assistant":
                    current_turn_messages.append({
                        "role": role,
                        "content": content,
                        "type": msg_type,
                        "message_index": i
                    })
            
            # Add the last turn
            if current_turn_messages:
                turn_number += 1
                turns.append({
                    "turn_number": turn_number,
                    "messages": current_turn_messages
                })
        
        conversation["turns"] = turns
        conversation["actions"] = action_sequence
        conversation["total_turns"] = len(turns)
        
        # Extract metadata
        conversation["metadata"] = {
            "total_actions": len(action_sequence),
            "total_messages": len(chat_history) if conversation["state_history"] else 0,
            "extracted_at": datetime.now().isoformat()
        }
        
        return conversation
    
    def extract_all_conversations(self, project_id: Optional[str] = None) -> List[Dict]:
        """Extract all conversations from storage."""
        conversations = []
        
        projects = [project_id] if project_id else self.list_projects()
        
        for proj_id in projects:
            apps = self.list_applications(proj_id)
            print(f"\nProject: {proj_id}")
            print(f"  Found {len(apps)} applications")
            
            for app in apps:
                conv = self.extract_conversation(proj_id, app["app_id"])
                if conv:
                    conversations.append(conv)
                    print(f"    - App {app['app_id'][:8]}...: {conv['total_turns']} turns")
        
        return conversations
    
    def save_to_json(self, conversations: List[Dict], output_file: str):
        """Save extracted conversations to a JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(conversations, f, indent=2)
        
        print(f"\n✓ Saved {len(conversations)} conversations to {output_file}")
    
    def generate_summary(self, conversations: List[Dict]) -> Dict:
        """Generate a summary of extracted conversations."""
        summary = {
            "total_conversations": len(conversations),
            "total_turns": sum(c["total_turns"] for c in conversations),
            "projects": defaultdict(int),
            "turn_distribution": defaultdict(int),
            "action_counts": defaultdict(int)
        }
        
        for conv in conversations:
            summary["projects"][conv["project_id"]] += 1
            summary["turn_distribution"][conv["total_turns"]] += 1
            
            for action in conv["actions"]:
                summary["action_counts"][action["action"]] += 1
        
        # Convert defaultdicts to regular dicts for JSON serialization
        summary["projects"] = dict(summary["projects"])
        summary["turn_distribution"] = dict(summary["turn_distribution"])
        summary["action_counts"] = dict(summary["action_counts"])
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract conversation data from Burr tracking files"
    )
    parser.add_argument(
        "--storage-dir",
        default="~/.burr",
        help="Path to Burr storage directory (default: ~/.burr)"
    )
    parser.add_argument(
        "--project-id",
        help="Specific project ID to extract (default: all projects)"
    )
    parser.add_argument(
        "--output",
        default="extracted_conversations.json",
        help="Output JSON file path (default: extracted_conversations.json)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Also generate a summary file"
    )
    parser.add_argument(
        "--list-projects",
        action="store_true",
        help="List all available projects and exit"
    )
    
    args = parser.parse_args()
    
    extractor = BurrConversationExtractor(args.storage_dir)
    
    if args.list_projects:
        projects = extractor.list_projects()
        print(f"\nFound {len(projects)} projects in {args.storage_dir}:")
        for proj in projects:
            apps = extractor.list_applications(proj)
            print(f"  - {proj}: {len(apps)} applications")
        return
    
    # Extract conversations
    print("Extracting conversations from Burr tracking files...")
    conversations = extractor.extract_all_conversations(args.project_id)
    
    if not conversations:
        print("No conversations found!")
        return
    
    # Save to JSON
    extractor.save_to_json(conversations, args.output)
    
    # Generate summary if requested
    if args.summary:
        summary = extractor.generate_summary(conversations)
        summary_file = args.output.replace(".json", "_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary to {summary_file}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total conversations: {summary['total_conversations']}")
        print(f"Total turns: {summary['total_turns']}")
        print(f"\nTurn distribution:")
        for turns, count in sorted(summary['turn_distribution'].items()):
            print(f"  {turns} turn(s): {count} conversation(s)")
        print(f"\nTop actions:")
        for action, count in sorted(summary['action_counts'].items(), 
                                     key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {action}: {count}")


if __name__ == "__main__":
    main()
