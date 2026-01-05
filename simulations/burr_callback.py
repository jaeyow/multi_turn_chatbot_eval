import sys
from pathlib import Path
from typing import List
from deepeval.simulator import ConversationSimulator
from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Turn

# Add parent directory to path to import application module
sys.path.insert(0, str(Path(__file__).parent.parent))
from application import application, TERMINAL_ACTIONS

# Cache Burr application instances by thread_id to maintain conversation state
_app_cache = {}

# Foundation for a multi-turn chatbot evaluation simulation using Burr
async def burr_model_callback(input: str, turns: List[Turn], thread_id: str) -> Turn:
    """
    This function wraps the Burr application so DeepEval can 'talk' to it.
    Reuses the same Burr app instance for all turns in the same conversation.
    """
    # Reuse existing app for this conversation or create new one
    if thread_id not in _app_cache:
        _app_cache[thread_id] = application(app_id=thread_id)
    
    app = _app_cache[thread_id] 
    
    _, streaming_container = await app.astream_result(
        halt_after=TERMINAL_ACTIONS, 
        inputs={"query": input}
    )
    
    # 3. Extract the response from Burr's state
    # This key ('response') depends on how you defined your Burr actions
    _, state = await streaming_container.get()
    bot_message = state["response"]["content"]
    
    return Turn(role="assistant", content=bot_message)
  
goldens = [
    ConversationalGolden(
        scenario="A user wants to drop by at 5:00 AM on a Saturday, to pick up an oline order.",
        expected_outcome="The bot should inform the user that the shop opens at 8:00 AM on Saturdays.",
        user_description="An early bird who is in a rush and quite persistent."
    ),
    ConversationalGolden(
        scenario="A user wants to check what if the store can fix their broken headphones.",
        expected_outcome="The bot should explain that they only fix bike-related items.",
        user_description="A confused customer who is not familiar with the store's services."
    )
]

def run_simulation():
    # Clear the cache before starting new simulations
    global _app_cache
    _app_cache = {}
    
    simulator = ConversationSimulator(
        model_callback=burr_model_callback
    )
    
    # Simulate the conversations (this handles async internally)
    test_cases = simulator.simulate(
        conversational_goldens=goldens,
        max_user_simulations=5  # Limit to 5 user-assistant exchanges per conversation
    )
    
    # Save these to your "Trace Library" for Analysis
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Simulation {i+1} Results ---")
        print(f"Scenario: {test_case.scenario}")
        print(f"Expected Outcome: {test_case.expected_outcome}")
        print(f"User Description: {test_case.user_description}")
        print(f"Total turns: {len(test_case.turns)}")
        print("\nConversation:")
        for turn in test_case.turns:
            print(f"  {turn.role}: {turn.content}")

if __name__ == "__main__":
    run_simulation()  