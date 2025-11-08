import asyncio
import copy
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


@action(reads=[], writes=["chat_history", "query"])
def process_query(state: State, query: str) -> Tuple[dict, State]:
    result = {"chat_item": {"role": "user", "content": query, "type": "text"}}
    return result, state.wipe(keep=["query", "chat_history"]).append(
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
        "- If the prompt is something along the lines of 'what bikes do you have?' or 'do you have mountain bikes in stock?', the mode would be 'product_inquiry'.\n"
        "- If the prompt is something along the lines of 'I need to book a service' or 'can I make an appointment?', the mode would be 'book_appointment'.\n"
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
        book_appointment=chat_response.bind(
            prepend_prompt="Please assist the customer with booking a service appointment",
        ),
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
        ("check_safety", "decide_mode", when(safe=True)),
        ("check_safety", "unsafe_response", default),
        ("decide_mode", "shop_info", when(mode="shop_info")),
        ("decide_mode", "product_inquiry", when(mode="product_inquiry")),
        ("decide_mode", "book_appointment", when(mode="book_appointment")),
        ("decide_mode", "maintenance_tips", when(mode="maintenance_tips")),
        ("decide_mode", "policy_question", when(mode="policy_question")),
        ("decide_mode", "what_can_you_do", when(mode="what_can_you_do")),
        ("decide_mode", "prompt_for_more", default),
        (
            [
                "shop_info",
                "product_inquiry",
                "book_appointment",
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
        .with_state(chat_history=[])
        .with_graph(graph)
        .with_tracker(project="demo_chatbot_streaming")
        .with_identifiers(app_id=app_id)
        .build()
    )


# TODO -- replace these with action tags when we have the availability
TERMINAL_ACTIONS = [
    "shop_info",
    "product_inquiry",
    "book_appointment",
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
