import asyncio
import copy
import json
from typing import AsyncGenerator, Optional, Tuple

import openai

from burr.core import ApplicationBuilder, State, default, when
from burr.core.action import action, streaming_action
from burr.core.graph import GraphBuilder
from dotenv import load_dotenv

load_dotenv()

MODES = [
    "shop_info",  # Hours, location, contact info
    "product_inquiry",  # Available bikes, accessories, availability
    "book_appointment",  # Service appointment booking
    "maintenance_tips",  # Bike maintenance advice
    "policy_question",  # Returns, warranties, delivery FAQs
    "what_can_you_do",  # Explain chatbot capabilities
    "unknown",
]

# Appointment booking configuration
APPOINTMENT_REQUIRED_FIELDS = [
    "service_type",      # tune-up, repair, full service, etc.
    "preferred_date",    # customer's preferred date
    "preferred_time",    # morning, afternoon, specific time
]

APPOINTMENT_OPTIONAL_FIELDS = [
    "bike_details",      # make/model if available
    "specific_issues",   # any specific problems to address
    "contact_info",      # phone/email for confirmation
]


@action(reads=[], writes=["chat_history", "query"])
def process_query(state: State, query: str) -> Tuple[dict, State]:
    result = {"chat_item": {"role": "user", "content": query, "type": "text"}}
    # Always preserve in_appointment_flow, appointment_data, and awaiting_confirmation
    keep_fields = ["query", "chat_history", "in_appointment_flow", "appointment_data", "awaiting_confirmation"]
    
    return result, state.wipe(keep=keep_fields).append(
        chat_history=result["chat_item"]
    ).update(query=query)


@action(reads=["query"], writes=["safe"])
def check_safety(state: State) -> Tuple[dict, State]:
    result = {"safe": "unsafe" not in state["query"]}  # quick hack to demonstrate
    return result, state.update(safe=result["safe"])


def _get_openai_client():
    return openai.AsyncOpenAI()


@action(reads=["query"], writes=["mode"])
async def choose_mode(state: State) -> Tuple[dict, State]:
    prompt = (
        f"You are a chatbot for JO's Bike Shop. You've been prompted this: {state['query']}. "
        f"You have the capability of responding in the following modes: {', '.join(MODES)}. "
        "Please respond with *only* a single word representing the mode that most accurately "
        "corresponds to the prompt. For instance:\n"
        "- If the prompt is something along the lines of 'what are your opening hours?' or 'where are you located?', the mode would be 'shop_info'.\n"
        "- If the prompt is something along the lines of 'what bikes do you have for sale?' or 'do you have mountain bikes in stock?', the mode would be 'product_inquiry'.\n"
        "- If the prompt is something along the lines of 'I need to book a service, or repair' or 'can I make an appointment?', the mode would be 'book_appointment'.\n"
        "- If the prompt is something along the lines of 'how do I maintain my bike?' or 'chain maintenance tips?', the mode would be 'maintenance_tips'.\n"
        "- If the prompt is something along the lines of 'what is your return policy?' or 'do you offer warranties?', the mode would be 'policy_question'.\n"
        "- If the customer is asking about what the chatbot can help with, the mode should be 'what_can_you_do'.\n"
        "If none of these modes apply, please respond with 'unknown'."
    )

    result = await _get_openai_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
    )
    content = result.choices[0].message.content
    mode = content.lower()
    if mode not in MODES:
        mode = "unknown"
    result = {"mode": mode}
    return result, state.update(**result)


@streaming_action(reads=["query", "chat_history"], writes=["response"])
async def prompt_for_more(
    state: State,
) -> AsyncGenerator[Tuple[dict, Optional[State]], None]:
    """Not streaming, as we have the result immediately."""
    result = {
        "response": {
            "content": "None of the response modes I support apply to your question. Please clarify?",
            "type": "text",
            "role": "assistant",
        }
    }
    for word in result["response"]["content"].split():
        await asyncio.sleep(0.1)
        yield {"delta": word + " "}, None
    yield result, state.update(**result).append(chat_history=result["response"])


@streaming_action(reads=["query", "chat_history", "mode"], writes=["response"])
async def chat_response(
    state: State, prepend_prompt: str, model: str = "gpt-3.5-turbo"
) -> AsyncGenerator[Tuple[dict, Optional[State]], None]:
    """Streaming action, as we don't have the result immediately. This makes it more interactive"""
    chat_history = copy.deepcopy(state["chat_history"])
    chat_history[-1]["content"] = f"{prepend_prompt}: {chat_history[-1]['content']}"
    chat_history_api_format = [
        {
            "role": chat["role"],
            "content": chat["content"],
        }
        for chat in chat_history
    ]
    client = _get_openai_client()
    result = await client.chat.completions.create(
        model=model, messages=chat_history_api_format, stream=True
    )
    buffer = []
    async for chunk in result:
        chunk_str = chunk.choices[0].delta.content
        if chunk_str is None:
            continue
        buffer.append(chunk_str)
        yield {
            "delta": chunk_str,
        }, None

    result = {
        "response": {"content": "".join(buffer), "type": "text", "role": "assistant"},
        "modified_chat_history": chat_history,
    }
    yield result, state.update(**result).append(chat_history=result["response"])


@streaming_action(reads=["query", "chat_history"], writes=["response"])
async def unsafe_response(state: State) -> AsyncGenerator[Tuple[dict, Optional[State]], None]:
    result = {
        "response": {
            "content": "I am afraid I can't respond to that...",
            "type": "text",
            "role": "assistant",
        }
    }
    for word in result["response"]["content"].split():
        await asyncio.sleep(0.1)
        yield {"delta": word + " "}, None
    yield result, state.update(**result).append(chat_history=result["response"])


@streaming_action(reads=["query", "chat_history"], writes=["response"])
async def explain_capabilities(
    state: State,
) -> AsyncGenerator[Tuple[dict, Optional[State]], None]:
    """Explains what the chatbot can help with."""
    result = {
        "response": {
            "content": (
                "I'm here to help you with JO's Bike Shop! I can assist you with:\n\n"
                "• Shop Information - Opening hours, location, and contact details\n"
                "• Product Inquiries - Available bikes, accessories, and product availability\n"
                "• Service Appointments - Booking bike service and repairs\n"
                "• Maintenance Tips - Advice on keeping your bike in top condition\n"
                "• Shop Policies - Questions about returns, warranties, and delivery\n\n"
                "How can I help you today?"
            ),
            "type": "text",
            "role": "assistant",
        }
    }
    for word in result["response"]["content"].split():
        await asyncio.sleep(0.05)
        yield {"delta": word + " "}, None
    yield result, state.update(**result).append(chat_history=result["response"])


async def _extract_appointment_info(query: str, chat_history: list) -> dict:
    """Extract ALL appointment information from query and context using LLM."""
    
    # Build context from recent chat history
    context = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in chat_history[-5:]  # last 5 messages for context
    ])
    
    prompt = f"""
Extract appointment booking information from this conversation:

{context}

Latest customer message: "{query}"

Extract any of these fields if mentioned:
- service_type: (tune-up, repair, full service, custom work, etc.)
- preferred_date: (specific date or relative like "next Tuesday", "this Friday")
- preferred_time: (morning, afternoon, specific time)
- bike_details: (brand, model, type of bike)
- specific_issues: (any problems they mention)
- contact_info: (phone or email if provided)

Return ONLY a JSON object with the fields you found. Use null for fields not mentioned.
Example: {{"service_type": "tune-up", "preferred_date": "next Tuesday", "preferred_time": "afternoon", "bike_details": "Trek mountain bike", "specific_issues": "clicking noise in gears", "contact_info": null}}

If nothing is found, return: {{}}
"""
    
    client = _get_openai_client()
    result = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise information extraction assistant. Extract structured data from conversation."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    
    extracted = json.loads(result.choices[0].message.content)
    
    # Filter out null values
    return {k: v for k, v in extracted.items() if v is not None}


def _ask_for_missing_info(missing_fields: list, current_data: dict) -> str:
    """Generate a natural question asking for missing required information."""
    
    # Acknowledge what we have so far if anything was collected
    if current_data:
        acknowledgment = "Great! "
        if "service_type" in current_data:
            acknowledgment += f"I have you down for a {current_data['service_type']}. "
    else:
        acknowledgment = ""
    
    # Ask for the first missing field
    field = missing_fields[0]
    
    questions = {
        "service_type": "What type of service do you need? We offer tune-ups, repairs, full services, and custom work.",
        "preferred_date": "What date works best for you?",
        "preferred_time": "What time would you prefer? We have morning (8am-12pm) and afternoon (1pm-5pm) slots available.",
    }
    
    return acknowledgment + questions.get(field, f"Could you provide your {field.replace('_', ' ')}?")


def _generate_appointment_confirmation(appointment_data: dict) -> str:
    """Generate confirmation message with all collected details."""
    
    message = "Perfect! Let me confirm your appointment:\n\n"
    
    field_labels = {
        "service_type": "Service",
        "preferred_date": "Date",
        "preferred_time": "Time",
        "bike_details": "Bike",
        "specific_issues": "Issues",
        "contact_info": "Contact",
    }
    
    for field in APPOINTMENT_REQUIRED_FIELDS + APPOINTMENT_OPTIONAL_FIELDS:
        if field in appointment_data and appointment_data[field]:
            label = field_labels.get(field, field.replace("_", " ").title())
            message += f"• {label}: {appointment_data[field]}\n"
    
    message += "\nDoes this look correct? I can book this for you now!"
    
    return message


async def _is_off_topic_query(query: str, chat_history: list) -> bool:
    """Detect if customer's query is off-topic (not related to appointment booking)."""
    
    prompt = f"""You are analyzing a customer query during an active appointment booking process.

Current query: "{query}"

Recent conversation context: {json.dumps(chat_history[-3:] if len(chat_history) > 3 else chat_history)}

Is this query related to booking/scheduling an appointment, or is it an off-topic question (like asking about products, shop hours, policies, etc.)?

Respond with ONLY "on_topic" or "off_topic".

Examples:
- "What are your hours?" -> off_topic
- "Do you sell helmets?" -> off_topic
- "Tuesday afternoon works" -> on_topic
- "I prefer morning" -> on_topic
- "Actually, I need a repair" -> on_topic
- "How much does a tune-up cost?" -> off_topic (price question, not scheduling)
- "Can I bring my bike in Thursday?" -> on_topic
"""
    
    client = _get_openai_client()
    result = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a query classifier. Respond only with 'on_topic' or 'off_topic'."},
            {"role": "user", "content": prompt},
        ],
    )
    
    response = result.choices[0].message.content.strip().lower()
    return response == "off_topic"


async def _classify_confirmation_response(query: str) -> str:
    """Classify customer's response to appointment confirmation.
    
    Returns:
        - "affirmative": Customer confirms/approves the appointment
        - "negative": Customer rejects/declines the appointment
        - "change": Customer wants to modify something
    """
    
    prompt = f"""You are analyzing a customer's response to an appointment confirmation.

Customer's response: "{query}"

Classify this response into one of these categories:
- "affirmative": Customer is confirming/approving the appointment (yes, correct, looks good, confirm it, book it, perfect, that works, etc.)
- "negative": Customer is rejecting/declining the appointment (no, that's wrong, not correct, cancel, nevermind, etc.)
- "change": Customer wants to modify something (change the time, different date, update the service, I'd like to change X, actually can we do Y instead, etc.)

Respond with ONLY one word: "affirmative", "negative", or "change".

Examples:
- "yes please" -> affirmative
- "looks good!" -> affirmative
- "perfect, book it" -> affirmative
- "that works for me" -> affirmative
- "no that doesn't work" -> negative
- "that's not right" -> negative
- "can we change the time?" -> change
- "I'd like to book for a different day" -> change
- "actually, make it morning instead" -> change
- "update the service to full service" -> change
"""
    
    client = _get_openai_client()
    result = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a response classifier. Respond only with 'affirmative', 'negative', or 'change'."},
            {"role": "user", "content": prompt},
        ],
    )
    
    response = result.choices[0].message.content.strip().lower()
    # Ensure we return a valid response
    if response not in ["affirmative", "negative", "change"]:
        return "change"  # Default to change if unclear
    return response


@streaming_action(reads=["query", "chat_history", "appointment_data"], writes=["response", "appointment_data", "in_appointment_flow", "awaiting_confirmation"])
async def start_appointment_booking(
    state: State,
) -> AsyncGenerator[Tuple[dict, Optional[State]], None]:
    """Initial appointment booking - extracts all available info and routes to multi-turn if needed."""
    
    # Extract information from initial query
    extracted_info = await _extract_appointment_info(
        state["query"], 
        state["chat_history"]
    )
    
    # Check what's still missing
    missing_required = [
        field for field in APPOINTMENT_REQUIRED_FIELDS 
        if field not in extracted_info or not extracted_info[field]
    ]
    
    if not missing_required:
        # Customer provided everything! Show confirmation immediately
        confirmation = _generate_appointment_confirmation(extracted_info)
        
        for word in confirmation.split():
            await asyncio.sleep(0.05)
            yield {"delta": word + " "}, None
        
        result = {
            "response": {
                "content": confirmation,
                "type": "text",
                "role": "assistant",
            },
            "appointment_data": extracted_info,  # Keep data until confirmed
            "in_appointment_flow": True,  # Enter appointment flow
            "awaiting_confirmation": True,  # Now waiting for confirmation
        }
        yield result, state.update(**result).append(chat_history=result["response"])
    else:
        # Missing info - ask for the first missing field and enter multi-turn flow
        next_question = _ask_for_missing_info(missing_required, extracted_info)
        
        for word in next_question.split():
            await asyncio.sleep(0.05)
            yield {"delta": word + " "}, None
        
        result = {
            "response": {
                "content": next_question,
                "type": "text",
                "role": "assistant",
            },
            "appointment_data": extracted_info,
            "in_appointment_flow": True,  # Enter appointment flow
            "awaiting_confirmation": False,
        }
        yield result, state.update(**result).append(chat_history=result["response"])


@streaming_action(reads=["query", "chat_history", "appointment_data", "awaiting_confirmation"], writes=["response", "appointment_data", "appointment_complete", "in_appointment_flow", "awaiting_confirmation"])
async def continue_appointment_booking(
    state: State,
) -> AsyncGenerator[Tuple[dict, Optional[State]], None]:
    """Multi-turn appointment booking - collects missing info and handles confirmation."""
    
    # Get existing appointment data
    appointment_data = state.get("appointment_data", {})
    awaiting_confirmation = state.get("awaiting_confirmation", False)
    
    # If we're awaiting confirmation, check customer's response
    if awaiting_confirmation:
        # Use LLM to classify the response
        response_type = await _classify_confirmation_response(state["query"])
        
        if response_type == "affirmative":
            # Customer confirmed! Book the appointment
            confirmation_message = "Excellent! Your appointment has been booked. We'll see you then! Is there anything else I can help you with?"
            
            for word in confirmation_message.split():
                await asyncio.sleep(0.05)
                yield {"delta": word + " "}, None
            
            result = {
                "response": {
                    "content": confirmation_message,
                    "type": "text",
                    "role": "assistant",
                },
                "appointment_data": {},  # Clear appointment data after completion
                "appointment_complete": True,
                "in_appointment_flow": False,  # Exit appointment flow
                "awaiting_confirmation": False,
            }
            yield result, state.update(**result).append(chat_history=result["response"])
            return
        
        elif response_type == "negative":
            # Customer rejected the appointment
            rejection_message = "I understand. I've canceled this appointment booking. Would you like to start over or can I help you with something else?"
            
            for word in rejection_message.split():
                await asyncio.sleep(0.05)
                yield {"delta": word + " "}, None
            
            result = {
                "response": {
                    "content": rejection_message,
                    "type": "text",
                    "role": "assistant",
                },
                "appointment_data": {},  # Clear appointment data
                "appointment_complete": False,
                "in_appointment_flow": False,  # Exit appointment flow
                "awaiting_confirmation": False,
            }
            yield result, state.update(**result).append(chat_history=result["response"])
            return
        
        else:  # response_type == "change"
            # Customer wants to make changes
            change_message = "No problem! What would you like to change?"
            
            for word in change_message.split():
                await asyncio.sleep(0.05)
                yield {"delta": word + " "}, None
            
            result = {
                "response": {
                    "content": change_message,
                    "type": "text",
                    "role": "assistant",
                },
                "appointment_data": appointment_data,  # Keep existing data
                "appointment_complete": False,
                "in_appointment_flow": True,  # Stay in appointment flow
                "awaiting_confirmation": False,  # Go back to collecting info
            }
            yield result, state.update(**result).append(chat_history=result["response"])
            return
    
    # Check if customer is asking an off-topic question
    query_lower = state["query"].lower()
    is_canceling = any(word in query_lower for word in ["cancel", "nevermind", "never mind", "forget it", "stop", "exit"])
    
    if is_canceling:
        # Customer wants to cancel the booking
        cancel_message = "No problem! I've canceled the appointment booking. Is there anything else I can help you with?"
        
        for word in cancel_message.split():
            await asyncio.sleep(0.05)
            yield {"delta": word + " "}, None
        
        result = {
            "response": {
                "content": cancel_message,
                "type": "text",
                "role": "assistant",
            },
            "appointment_data": {},  # Clear appointment data
            "appointment_complete": False,
            "in_appointment_flow": False,  # Exit appointment flow
            "awaiting_confirmation": False,
        }
        yield result, state.update(**result).append(chat_history=result["response"])
        return
    
    # Check if query is off-topic (customer asking about something else)
    is_off_topic = await _is_off_topic_query(state["query"], state["chat_history"])
    
    if is_off_topic:
        # Gently redirect back to appointment booking
        missing_required = [
            field for field in APPOINTMENT_REQUIRED_FIELDS 
            if field not in appointment_data or not appointment_data[field]
        ]
        
        if missing_required:
            redirect_message = (
                "I'd be happy to help with that, but first let's finish booking your appointment! "
                + _ask_for_missing_info(missing_required, appointment_data)
                + " (Or say 'cancel' if you'd like to stop the booking.)"
            )
        else:
            redirect_message = "I can help with that question, but first let's confirm your appointment details. Does the information I shared look correct?"
        
        for word in redirect_message.split():
            await asyncio.sleep(0.05)
            yield {"delta": word + " "}, None
        
        result = {
            "response": {
                "content": redirect_message,
                "type": "text",
                "role": "assistant",
            },
            "appointment_data": appointment_data,  # Keep existing data
            "appointment_complete": False,
            "in_appointment_flow": True,  # Stay in appointment flow
            "awaiting_confirmation": awaiting_confirmation,  # Keep confirmation state
        }
        yield result, state.update(**result).append(chat_history=result["response"])
        return
    
    # Extract new information from current query
    extracted_info = await _extract_appointment_info(
        state["query"], 
        state["chat_history"]
    )
    
    # Merge with existing data (new info takes precedence)
    appointment_data.update(extracted_info)
    
    # Check what's still missing
    missing_required = [
        field for field in APPOINTMENT_REQUIRED_FIELDS 
        if field not in appointment_data or not appointment_data[field]
    ]
    
    if not missing_required:
        # We have everything! Generate confirmation and ask for approval
        confirmation = _generate_appointment_confirmation(appointment_data)
        
        for word in confirmation.split():
            await asyncio.sleep(0.05)
            yield {"delta": word + " "}, None
        
        result = {
            "response": {
                "content": confirmation,
                "type": "text",
                "role": "assistant",
            },
            "appointment_data": appointment_data,  # Keep data until confirmed
            "appointment_complete": False,  # Not complete until confirmed
            "in_appointment_flow": True,  # Stay in appointment flow
            "awaiting_confirmation": True,  # Now waiting for confirmation
        }
        yield result, state.update(**result).append(chat_history=result["response"])
    else:
        # Ask for missing information and stay in appointment flow
        next_question = _ask_for_missing_info(missing_required, appointment_data)
        
        for word in next_question.split():
            await asyncio.sleep(0.05)
            yield {"delta": word + " "}, None
        
        result = {
            "response": {
                "content": next_question,
                "type": "text",
                "role": "assistant",
            },
            "appointment_data": appointment_data,
            "appointment_complete": False,
            "in_appointment_flow": True,  # Stay in appointment flow
            "awaiting_confirmation": False,
        }
        yield result, state.update(**result).append(chat_history=result["response"])


graph = (
    GraphBuilder()
    .with_actions(
        query=process_query,
        check_safety=check_safety,
        unsafe_response=unsafe_response,
        decide_mode=choose_mode,
        shop_info=chat_response.bind(
            prepend_prompt="Please provide information about JO's Bike Shop (hours, location, contact) based on the following query",
        ),
        product_inquiry=chat_response.bind(
            prepend_prompt="Please help the customer with their product inquiry about bikes or accessories",
        ),
        start_booking=start_appointment_booking,
        continue_booking=continue_appointment_booking,
        maintenance_tips=chat_response.bind(
            prepend_prompt="Please provide helpful bike maintenance tips for the following",
        ),
        policy_question=chat_response.bind(
            prepend_prompt="Please answer the customer's question about shop policies (returns, warranties, delivery)",
        ),
        what_can_you_do=explain_capabilities,
        prompt_for_more=prompt_for_more,
    )
    .with_transitions(
        ("query", "check_safety", default),
        ("check_safety", "continue_booking", when(safe=True, in_appointment_flow=True)),
        ("check_safety", "decide_mode", when(safe=True)),
        ("check_safety", "unsafe_response", default),
        ("decide_mode", "shop_info", when(mode="shop_info")),
        ("decide_mode", "product_inquiry", when(mode="product_inquiry")),
        ("decide_mode", "start_booking", when(mode="book_appointment")),
        ("decide_mode", "maintenance_tips", when(mode="maintenance_tips")),
        ("decide_mode", "policy_question", when(mode="policy_question")),
        ("decide_mode", "what_can_you_do", when(mode="what_can_you_do")),
        ("decide_mode", "prompt_for_more", default),
        (
            [
                "shop_info",
                "product_inquiry",
                "start_booking",
                "continue_booking",
                "maintenance_tips",
                "policy_question",
                "what_can_you_do",
                "prompt_for_more",
                "unsafe_response",
            ],
            "query",
        ),
    )
    .build()
)


def application(app_id: Optional[str] = None):
    return (
        ApplicationBuilder()
        .with_entrypoint("query")
        .with_state(chat_history=[], appointment_data={}, in_appointment_flow=False, awaiting_confirmation=False)
        .with_graph(graph)
        .with_tracker(project="multi_turn_chatbot_eval")
        .with_identifiers(app_id=app_id)
        .build()
    )


# TODO -- replace these with action tags when we have the availability
TERMINAL_ACTIONS = [
    "shop_info",
    "product_inquiry",
    "start_booking",
    "continue_booking",
    "maintenance_tips",
    "policy_question",
    "what_can_you_do",
    "prompt_for_more",
    "unsafe_response",
]
if __name__ == "__main__":
    app = application()
    app.visualize(
        output_file_path="statemachine",
        include_conditions=True,
        view=True,
        format="png",
    )
