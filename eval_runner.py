"""
End-to-end evaluation runner for JO's Bike Shop chatbot.

Runs all single-turn and multi-turn evaluations using DeepEval metrics plus
LLM-assisted open coding and failure mode taxonomy analysis. Overwrites the
baseline result files in error_analysis/.

Usage:
    python eval_runner.py
    python eval_runner.py --judge-model gpt-4o
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress noisy third-party warnings before any imports trigger them
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message="Overriding of current TracerProvider")
warnings.filterwarnings("ignore", message="Task was destroyed but it is pending")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="coroutine.*was never awaited")
os.environ.setdefault("DEEPEVAL_VERBOSE_MODE", "NO")  # suppress "✨ running..." lines

import logging
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

from dotenv import load_dotenv
from openai import OpenAI

from deepeval.dataset import ConversationalGolden
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ConversationCompletenessMetric,
    GEval,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
    TaskCompletionMetric,
)
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
    Turn,
)

from application import application, TERMINAL_ACTIONS, SHOP_INFO

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_JUDGE_MODEL = "gpt-4o"
ERROR_ANALYSIS_DIR = Path("error_analysis")
CACHE_DIR = ERROR_ANALYSIS_DIR / ".cache"

# GPT-4o pricing (USD per 1M tokens) — update when pricing changes
_PRICING: Dict[str, Tuple[float, float]] = {
    "gpt-4o":       (2.50, 10.00),
    "gpt-4o-mini":  (0.15,  0.60),
    "gpt-4-turbo":  (10.00, 30.00),
}
_DEFAULT_PRICE = (2.50, 10.00)  # fallback if model not in table


@dataclass
class CostTracker:
    """Accumulates token usage and cost estimates across the full eval run."""
    bot_model: str = "gpt-4o"
    judge_model: str = "gpt-4o"

    # Bot (chatbot under test)
    bot_input_tokens: int = 0
    bot_output_tokens: int = 0

    # Judge — DeepEval metrics (tracked via metric.evaluation_cost)
    deepeval_cost_usd: float = 0.0

    # Judge — direct OpenAI calls (open coding + failure mode analysis)
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0

    def _price(self, model: str) -> Tuple[float, float]:
        return _PRICING.get(model, _DEFAULT_PRICE)

    def record_bot_exchange(self, query: str, response: str) -> None:
        """Estimate bot tokens from string length (4 chars ≈ 1 token)."""
        self.bot_input_tokens += len(query) // 4
        self.bot_output_tokens += len(response) // 4

    def record_deepeval_metric(self, metric: Any) -> None:
        self.deepeval_cost_usd += getattr(metric, "evaluation_cost", 0) or 0.0

    def record_judge_usage(self, usage: Any) -> None:
        """Record usage from an openai ChatCompletion response."""
        self.judge_input_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.judge_output_tokens += getattr(usage, "completion_tokens", 0) or 0

    @property
    def bot_cost_usd(self) -> float:
        p_in, p_out = self._price(self.bot_model)
        return (self.bot_input_tokens / 1_000_000 * p_in
                + self.bot_output_tokens / 1_000_000 * p_out)

    @property
    def judge_direct_cost_usd(self) -> float:
        p_in, p_out = self._price(self.judge_model)
        return (self.judge_input_tokens / 1_000_000 * p_in
                + self.judge_output_tokens / 1_000_000 * p_out)

    @property
    def judge_cost_usd(self) -> float:
        return self.deepeval_cost_usd + self.judge_direct_cost_usd

    @property
    def total_cost_usd(self) -> float:
        return self.bot_cost_usd + self.judge_cost_usd

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "bot": {
                "model": self.bot_model,
                "input_tokens_estimated": self.bot_input_tokens,
                "output_tokens_estimated": self.bot_output_tokens,
                "cost_usd_estimated": round(self.bot_cost_usd, 6),
            },
            "judge": {
                "model": self.judge_model,
                "deepeval_metrics_cost_usd": round(self.deepeval_cost_usd, 6),
                "direct_calls_input_tokens": self.judge_input_tokens,
                "direct_calls_output_tokens": self.judge_output_tokens,
                "direct_calls_cost_usd": round(self.judge_direct_cost_usd, 6),
                "total_judge_cost_usd": round(self.judge_cost_usd, 6),
            },
            "total_cost_usd": round(self.total_cost_usd, 6),
            "note": (
                "Bot token counts are estimated (len/4). "
                "Judge DeepEval costs are exact per DeepEval; "
                "direct call costs are exact from OpenAI usage responses."
            ),
        }

CHATBOT_ROLE = (
    "A helpful customer service chatbot for JO's Bike Shop in Portland, OR. "
    "It books service appointments (collecting service type, date, and time), "
    "answers shop hours/location/contact questions, provides product information, "
    "gives bike maintenance advice, and answers policy questions."
)

SHOP_INFO_STR = json.dumps(SHOP_INFO, indent=2)

# Pre-defined failure mode taxonomy (from error_analysis_plan.md)
FAILURE_MODES = {
    "FM1_intent_detection_error": "Bot misclassified the user's intent and routed to the wrong handler",
    "FM2_info_extraction_issue": "Bot failed to extract information the user clearly provided (e.g. asked for service type when user already said 'tune-up')",
    "FM3_state_management_problem": "Appointment data or booking state was lost or corrupted across turns",
    "FM4_context_loss": "Bot failed to retain or reference information from earlier turns",
    "FM5_correction_handling_failure": "Bot did not handle a user correction gracefully (created a new booking instead of updating, etc.)",
    "FM6_confirmation_flow_issue": "Confirmation step was skipped, repeated, or handled incorrectly",
    "FM7_off_topic_redirection_failure": "Bot did not redirect a distracted user back to the active booking flow",
    "FM8_cancellation_not_detected": "Bot failed to recognise that the user wanted to cancel or abandon the interaction",
    "FM9_recall_failure": "Bot could not recall or correctly reference a booking made earlier in the session",
    "FM10_response_quality_issue": "Response was factually wrong, incoherent, too brief, or unhelpful",
    "FM11_redundant_questioning": "Bot asked for information the user had already provided",
    "FM12_flow_transition_error": "Bot transitioned between states incorrectly (e.g. stayed in booking flow after confirmation)",
}

# ─── Cache helpers (for --start-step resumption) ─────────────────────────────


def _save_cache(name: str, data: Any) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{name}.json").write_text(json.dumps(data, indent=2))


def _load_cache(name: str) -> Any:
    path = CACHE_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")
    return json.loads(path.read_text())


# Keys required to be cached before each step can be skipped
_STEP_CACHE_DEPS = {
    2: ["st_results"],
    3: ["st_results", "st_scores"],
    4: ["st_results", "st_scores", "mt_results"],
    5: ["st_results", "st_scores", "mt_results", "mt_scores"],
    6: ["st_results", "st_scores", "mt_results", "mt_scores",
        "st_open_coding", "mt_open_coding", "st_failure_modes", "mt_failure_modes"],
}


def _resolve_start_step(requested: int) -> int:
    """Return the earliest step we can actually start from given cached files."""
    for step in range(requested, 0, -1):
        deps = _STEP_CACHE_DEPS.get(step, [])
        if all((CACHE_DIR / f"{d}.json").exists() for d in deps):
            if step < requested:
                print(
                    f"  ⚠  Cache for step {requested} incomplete — "
                    f"falling back to step {step}."
                )
            return step
    return 1


def _rebuild_mt_test_cases(mt_results: List[Dict]) -> None:
    """Re-attach ConversationalTestCase objects to cached multi-turn results."""
    for r in mt_results:
        r["test_case"] = ConversationalTestCase(
            turns=[Turn(role=t["role"], content=t["content"]) for t in r["turns"]],
            chatbot_role=CHATBOT_ROLE,
        )


# ─── Bot helpers ──────────────────────────────────────────────────────────────


async def _query_bot(app, query: str, tracker: Optional["CostTracker"] = None):
    """Send one query to a Burr app instance; return (action_name, response_text)."""
    action_obj, streaming_container = await app.astream_result(
        halt_after=TERMINAL_ACTIONS,
        inputs={"query": query},
    )
    _, state = await streaming_container.get()
    response = state["response"]["content"]
    if tracker:
        tracker.record_bot_exchange(query, response)
    return action_obj.name, response


# ─── Single-turn ──────────────────────────────────────────────────────────────


async def run_single_turn_scenarios(examples: List[Dict], tracker: "CostTracker") -> List[Dict]:
    """Run all single-turn scenarios through the bot and collect raw traces."""
    results = []
    for i, ex in enumerate(examples):
        test_id = f"ST{i+1:03d}"
        query = ex["user_query"]
        print(f"  [{test_id}] {query}")
        app = application(app_id=test_id)
        action_name, response = await _query_bot(app, query, tracker)
        results.append(
            {
                "test_id": test_id,
                "scenario": ex["scenario"],
                "user_query": query,
                "bot_response": response,
                "action_taken": action_name,
                "expected_mode": ex["expected_mode"],
                "mode_correct": action_name == ex["expected_mode"],
                "success": True,
                "error": None,
            }
        )
    return results


def build_single_turn_test_cases(results: List[Dict]) -> List[LLMTestCase]:
    return [
        LLMTestCase(
            input=r["user_query"],
            actual_output=r["bot_response"],
        )
        for r in results
    ]


async def run_single_turn_metrics(
    test_cases: List[LLMTestCase], judge_model: str, tracker: "CostTracker"
) -> List[Dict]:
    """Run DeepEval metrics on single-turn test cases; return per-case scores."""
    # Build named metric factories so we can create a fresh instance per test case
    # (metric objects are stateful — reusing across test cases gives wrong scores)
    def make_metrics() -> List[Tuple[str, Any]]:
        return [
            ("Answer Relevancy", AnswerRelevancyMetric(model=judge_model, include_reason=True)),
            ("Task Completion", GEval(
                name="Task Completion",
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                criteria=(
                    "Does the response completely fulfil the user's request? "
                    "For shop information queries it must include accurate hours, address, or contact details. "
                    "For maintenance tips it must provide at least two concrete, actionable tips. "
                    "For policy questions it must address the specific policy asked about. "
                    "For capability questions it must list the chatbot's main functions."
                ),
                model=judge_model,
            )),
            ("Conversation Quality", GEval(
                name="Conversation Quality",
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                criteria=(
                    "Evaluate the response on three dimensions: "
                    "(1) Factual accuracy — shop details must match: "
                    f"address '{SHOP_INFO['address']['full']}', "
                    f"phone '{SHOP_INFO['contact']['phone']}', "
                    "hours Mon-Wed 9 AM-6 PM, Thu-Fri 9 AM-7 PM, Sat 8 AM-5 PM, Sun 10 AM-4 PM. "
                    "(2) Tone and naturalness — sounds like a friendly, professional shop assistant. "
                    "(3) Appropriate length — not needlessly verbose, not so short it omits key information."
                ),
                model=judge_model,
            )),
        ]

    scores = []
    for i, tc in enumerate(test_cases):
        print(f"  Scoring ST{i+1:03d} — {tc.input[:65]}...")
        entry: Dict[str, Any] = {}
        named_metrics = make_metrics()
        await asyncio.gather(*[m.a_measure(tc) for _, m in named_metrics])
        for name, m in named_metrics:
            entry[name] = {
                "score": getattr(m, "score", None),
                "reason": getattr(m, "reason", None),
            }
            tracker.record_deepeval_metric(m)
        scores.append(entry)
    return scores


# ─── Multi-turn ───────────────────────────────────────────────────────────────


def _scenario_to_golden(ex: Dict) -> ConversationalGolden:
    """Convert a synthetic multi-turn example to a ConversationalGolden."""
    sc = ex["scenario"]
    behavior_map = {
        "cooperative": "A cooperative user who answers questions directly and stays on topic.",
        "conversational": "A conversational user who uses natural, informal language and may include extra context.",
        "corrective": "A user who may change their mind or correct information they previously provided.",
        "distracted": "A user who goes off-topic during multi-turn flows but eventually returns to the main task.",
        "canceling": "A user who may decide to cancel or abandon the interaction mid-conversation.",
    }
    user_description = behavior_map.get(
        sc["user_behavior"],
        "A typical bike shop customer.",
    )

    scenario_desc = (
        f"A customer of JO's Bike Shop wants to: {ex['user_goal']}. "
        f"The conversation pattern is '{sc['interaction_pattern']}' "
        f"with '{sc['completeness']}' information provided. "
        f"Initial message: \"{ex['initial_query']}\""
    )

    expected_outcome = (
        f"The chatbot should help the user achieve their goal: {ex['user_goal']}. "
        f"Primary intent: {sc['primary_intent']}."
    )

    return ConversationalGolden(
        scenario=scenario_desc,
        expected_outcome=expected_outcome,
        user_description=user_description,
    )


async def simulate_one_multi_turn(
    golden: ConversationalGolden,
    test_id: str,
    judge_model: str,
    tracker: "CostTracker",
    max_turns: int = 8,
) -> ConversationalTestCase:
    """Simulate one multi-turn conversation using DeepEval ConversationSimulator."""
    app = application(app_id=test_id)

    async def bot_callback(input: str, **kwargs) -> Turn:  # noqa: A002
        _, response = await _query_bot(app, input, tracker)
        return Turn(role="assistant", content=response)

    simulator = ConversationSimulator(
        model_callback=bot_callback,
        simulator_model=judge_model,
        async_mode=True,
    )
    test_cases = simulator.simulate(
        conversational_goldens=[golden],
        max_user_simulations=max_turns,
    )
    tc = test_cases[0]
    tc.chatbot_role = CHATBOT_ROLE
    return tc


async def run_multi_turn_scenarios(
    examples: List[Dict], judge_model: str, tracker: "CostTracker"
) -> List[Dict]:
    """Simulate all multi-turn scenarios and collect raw traces + test cases."""
    results = []
    for i, ex in enumerate(examples):
        test_id = f"MT{i+1:03d}"
        print(f"  [{test_id}] {ex['scenario']['tuple']} — {ex['user_goal'][:50]}...")
        golden = _scenario_to_golden(ex)
        test_case = await simulate_one_multi_turn(golden, test_id, judge_model, tracker)
        turns = [
            {"role": t.role, "content": t.content} for t in test_case.turns
        ]
        results.append(
            {
                "test_id": test_id,
                "scenario": ex["scenario"],
                "user_goal": ex["user_goal"],
                "initial_query": ex["initial_query"],
                "turns": turns,
                "conversation": turns,
                "test_case": test_case,
                "success": True,
                "error": None,
            }
        )
    return results


async def run_multi_turn_metrics(
    results: List[Dict], judge_model: str, tracker: "CostTracker"
) -> List[Dict]:
    """Run DeepEval conversational metrics on multi-turn test cases."""
    def make_metrics() -> List[Tuple[str, Any]]:
        return [
            ("Conversation Completeness", ConversationCompletenessMetric(model=judge_model, include_reason=True)),
            ("Knowledge Retention", KnowledgeRetentionMetric(model=judge_model, include_reason=True)),
            ("Role Adherence", RoleAdherenceMetric(model=judge_model, include_reason=True)),
        ]

    scores = []
    for i, r in enumerate(results):
        tc = r["test_case"]
        print(f"  Scoring MT{i+1:03d} — {r['scenario']['tuple'][:60]}...")
        entry: Dict[str, Any] = {}
        named_metrics = make_metrics()
        await asyncio.gather(*[m.a_measure(tc) for _, m in named_metrics])
        for name, m in named_metrics:
            entry[name] = {
                "score": getattr(m, "score", None),
                "reason": getattr(m, "reason", None),
            }
            tracker.record_deepeval_metric(m)
        scores.append(entry)
    return scores


# ─── Open coding ─────────────────────────────────────────────────────────────


def _format_conversation_for_coding(result: Dict, is_multi_turn: bool) -> str:
    if is_multi_turn:
        turns = result.get("turns", [])
        lines = [f"  {t['role'].upper()}: {t['content']}" for t in turns]
        conv_text = "\n".join(lines)
        return (
            f"User goal: {result['user_goal']}\n"
            f"Scenario: {result['scenario']['tuple']}\n\n"
            f"Conversation:\n{conv_text}"
        )
    else:
        return (
            f"User query: {result['user_query']}\n"
            f"Expected mode: {result['expected_mode']}\n"
            f"Action taken: {result['action_taken']}\n"
            f"Bot response: {result['bot_response']}"
        )


OPEN_CODING_SYSTEM = (
    "You are an expert evaluator analysing chatbot conversation traces for quality issues. "
    "Be specific and concise. Focus on what actually happened in the trace, not hypotheticals."
)

OPEN_CODING_USER_TEMPLATE = """Review this chatbot conversation trace and provide open coding notes.

{conversation}

Provide your analysis in this exact JSON format:
{{
  "what_worked": "1-2 sentences on what the bot did well",
  "what_went_wrong": "1-2 sentences on any failures, errors, or missed opportunities (write 'None' if everything was correct)",
  "notable_behaviors": "any interesting patterns — missed intents, redundant questions, lost context, awkward phrasing, etc. (write 'None' if nothing notable)",
  "overall_success": 2
}}

For overall_success: 2 = complete success, 1 = partial success with minor issues, 0 = clear failure.

Respond with only the JSON object."""


def run_open_coding(
    results: List[Dict], is_multi_turn: bool, client: OpenAI,
    judge_model: str, tracker: "CostTracker"
) -> List[Dict]:
    """Use the judge LLM to generate open coding notes for each trace."""
    notes = []
    for result in results:
        conv_text = _format_conversation_for_coding(result, is_multi_turn)
        prompt = OPEN_CODING_USER_TEMPLATE.format(conversation=conv_text)
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": OPEN_CODING_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            tracker.record_judge_usage(response.usage)
            note = json.loads(response.choices[0].message.content)
        except Exception as e:
            note = {
                "what_worked": "Error during open coding",
                "what_went_wrong": str(e),
                "notable_behaviors": "None",
                "overall_success": 1,
            }
        notes.append(note)
    return notes


# ─── Failure mode analysis ────────────────────────────────────────────────────


FAILURE_MODE_SYSTEM = (
    "You are an expert chatbot evaluator. "
    "Classify a chatbot conversation trace into the provided failure mode taxonomy. "
    "A single trace can exhibit multiple failure modes. Be conservative — only flag a failure mode if there is clear evidence."
)

FAILURE_MODE_USER_TEMPLATE = """Analyse this chatbot conversation trace and classify it against the failure mode taxonomy.

{conversation}

Open coding notes: {notes}

Failure mode taxonomy:
{taxonomy}

Return a JSON object where each key is a failure mode ID and the value is 1 (present) or 0 (not present).
Example: {{"FM1_intent_detection_error": 0, "FM2_info_extraction_issue": 1, ...}}

Include all 12 failure mode keys. Respond with only the JSON object."""


def run_failure_mode_analysis(
    results: List[Dict],
    open_coding_notes: List[Dict],
    is_multi_turn: bool,
    client: OpenAI,
    judge_model: str,
    tracker: "CostTracker",
) -> List[Dict]:
    """Classify each trace into failure modes using the judge LLM."""
    taxonomy_str = "\n".join(
        f"  {k}: {v}" for k, v in FAILURE_MODES.items()
    )
    classifications = []
    for result, notes in zip(results, open_coding_notes):
        conv_text = _format_conversation_for_coding(result, is_multi_turn)
        notes_str = (
            f"What worked: {notes.get('what_worked', '')}\n"
            f"What went wrong: {notes.get('what_went_wrong', '')}\n"
            f"Notable behaviors: {notes.get('notable_behaviors', '')}"
        )
        prompt = FAILURE_MODE_USER_TEMPLATE.format(
            conversation=conv_text,
            notes=notes_str,
            taxonomy=taxonomy_str,
        )
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": FAILURE_MODE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            tracker.record_judge_usage(response.usage)
            classification = json.loads(response.choices[0].message.content)
            for k in FAILURE_MODES:
                classification.setdefault(k, 0)
        except Exception as e:
            classification = {k: 0 for k in FAILURE_MODES}
            classification["_error"] = str(e)
        classifications.append(classification)
    return classifications


def compute_failure_mode_frequencies(classifications: List[Dict]) -> Dict[str, int]:
    freq: Dict[str, int] = {k: 0 for k in FAILURE_MODES}
    for c in classifications:
        for k in FAILURE_MODES:
            freq[k] += c.get(k, 0)
    return freq


# ─── Saving results ───────────────────────────────────────────────────────────


def save_single_turn_results(
    results: List[Dict], scores: List[Dict], timestamp: str
):
    out = {
        "metadata": {
            "conversation_type": "single_turn",
            "description": "Single-Turn conversation tests",
            "timestamp": timestamp,
            "total_tests": len(results),
            "successful_tests": sum(1 for r in results if r["success"]),
        },
        "results": [
            {
                "test_id": r["test_id"],
                "scenario": r["scenario"],
                "user_query": r["user_query"],
                "bot_response": r["bot_response"],
                "action_taken": r["action_taken"],
                "expected_mode": r["expected_mode"],
                "mode_correct": r["mode_correct"],
                "success": r["success"],
                "error": r["error"],
            }
            for r in results
        ],
    }
    path = ERROR_ANALYSIS_DIR / "single_turn_results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  Saved → {path}")


def save_multi_turn_results(
    results: List[Dict], timestamp: str
):
    out = {
        "metadata": {
            "conversation_type": "multi_turn",
            "description": "Multi-Turn conversation tests",
            "timestamp": timestamp,
            "total_tests": len(results),
            "successful_tests": sum(1 for r in results if r["success"]),
        },
        "results": [
            {
                "test_id": r["test_id"],
                "scenario": r["scenario"],
                "user_goal": r["user_goal"],
                "initial_query": r["initial_query"],
                "turns": r["turns"],
                "conversation": r["conversation"],
                "success": r["success"],
                "error": r["error"],
            }
            for r in results
        ],
    }
    path = ERROR_ANALYSIS_DIR / "multi_turn_results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  Saved → {path}")


def save_single_turn_csv(
    results: List[Dict],
    scores: List[Dict],
    open_coding: List[Dict],
    failure_modes: List[Dict],
):
    path = ERROR_ANALYSIS_DIR / "single_turn_analysis.csv"
    fieldnames = [
        "Trace_ID", "Conversation_Type", "Scenario_Tuple",
        "Primary_Intent", "Completeness", "Interaction_Pattern", "User_Behavior",
        "User_Query", "Bot_Response_Preview", "Action_Taken", "Mode_Detected",
        "Mode_Correct", "Success", "Error",
        "Answer_Relevancy_Score", "Task_Completion_Score", "Conversation_Quality_Score",
        "Open_Code_What_Worked", "Open_Code_What_Went_Wrong", "Open_Code_Notable_Behaviors",
        "Overall_Success",
    ] + list(FAILURE_MODES.keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r, s, oc, fm in zip(results, scores, open_coding, failure_modes):
            sc = r["scenario"]
            row = {
                "Trace_ID": r["test_id"],
                "Conversation_Type": "single_turn",
                "Scenario_Tuple": sc["tuple"],
                "Primary_Intent": sc["primary_intent"],
                "Completeness": sc["completeness"],
                "Interaction_Pattern": sc["interaction_pattern"],
                "User_Behavior": sc["user_behavior"],
                "User_Query": r["user_query"],
                "Bot_Response_Preview": r["bot_response"][:100],
                "Action_Taken": r["action_taken"],
                "Mode_Detected": r["action_taken"],
                "Mode_Correct": r["mode_correct"],
                "Success": int(r["success"]),
                "Error": r["error"] or "",
                "Answer_Relevancy_Score": s.get("Answer Relevancy", {}).get("score", ""),
                "Task_Completion_Score": s.get("Task Completion", {}).get("score", ""),
                "Conversation_Quality_Score": s.get("Conversation Quality", {}).get("score", ""),
                "Open_Code_What_Worked": oc.get("what_worked", ""),
                "Open_Code_What_Went_Wrong": oc.get("what_went_wrong", ""),
                "Open_Code_Notable_Behaviors": oc.get("notable_behaviors", ""),
                "Overall_Success": oc.get("overall_success", ""),
            }
            for k in FAILURE_MODES:
                row[k] = fm.get(k, 0)
            writer.writerow(row)
    print(f"  Saved → {path}")


def save_multi_turn_csv(
    results: List[Dict],
    scores: List[Dict],
    open_coding: List[Dict],
    failure_modes: List[Dict],
):
    path = ERROR_ANALYSIS_DIR / "multi_turn_analysis.csv"
    fieldnames = [
        "Trace_ID", "Conversation_Type", "Scenario_Tuple",
        "Primary_Intent", "Completeness", "Interaction_Pattern", "User_Behavior",
        "User_Goal", "Num_Turns", "Success", "Error",
        "Conversation_Completeness_Score", "Knowledge_Retention_Score",
        "Role_Adherence_Score",
        "Open_Code_What_Worked", "Open_Code_What_Went_Wrong", "Open_Code_Notable_Behaviors",
        "Overall_Success",
    ] + list(FAILURE_MODES.keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r, s, oc, fm in zip(results, scores, open_coding, failure_modes):
            sc = r["scenario"]
            row = {
                "Trace_ID": r["test_id"],
                "Conversation_Type": "multi_turn",
                "Scenario_Tuple": sc["tuple"],
                "Primary_Intent": sc["primary_intent"],
                "Completeness": sc["completeness"],
                "Interaction_Pattern": sc["interaction_pattern"],
                "User_Behavior": sc["user_behavior"],
                "User_Goal": r["user_goal"],
                "Num_Turns": len(r["turns"]),
                "Success": int(r["success"]),
                "Error": r["error"] or "",
                "Conversation_Completeness_Score": s.get("Conversation Completeness", {}).get("score", ""),
                "Knowledge_Retention_Score": s.get("Knowledge Retention", {}).get("score", ""),
                "Role_Adherence_Score": s.get("Role Adherence", {}).get("score", ""),
                "Open_Code_What_Worked": oc.get("what_worked", ""),
                "Open_Code_What_Went_Wrong": oc.get("what_went_wrong", ""),
                "Open_Code_Notable_Behaviors": oc.get("notable_behaviors", ""),
                "Overall_Success": oc.get("overall_success", ""),
            }
            for k in FAILURE_MODES:
                row[k] = fm.get(k, 0)
            writer.writerow(row)
    print(f"  Saved → {path}")


def save_baseline_scores(
    st_results: List[Dict],
    st_scores: List[Dict],
    st_open_coding: List[Dict],
    st_failure_modes: List[Dict],
    mt_results: List[Dict],
    mt_scores: List[Dict],
    mt_open_coding: List[Dict],
    mt_failure_modes: List[Dict],
    judge_model: str,
    timestamp: str,
    tracker: "CostTracker",
):
    """Save machine-readable baseline snapshot for before/after model comparison."""

    def _avg(lst, key):
        vals = [s.get(key, {}).get("score") for s in lst if s.get(key, {}).get("score") is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    baseline = {
        "metadata": {
            "timestamp": timestamp,
            "judge_model": judge_model,
            "chatbot_model": "gpt-4o",
            "description": "Baseline evaluation scores before LLM replacement",
            "cost": tracker.summary_dict(),
        },
        "single_turn": {
            "total_scenarios": len(st_results),
            "mode_detection_accuracy": round(
                sum(1 for r in st_results if r["mode_correct"]) / len(st_results), 4
            ) if st_results else None,
            "avg_answer_relevancy": _avg(st_scores, "Answer Relevancy"),
            "avg_task_completion": _avg(st_scores, "Task Completion"),
            "avg_conversation_quality": _avg(st_scores, "Conversation Quality"),
            "overall_success_distribution": {
                str(v): sum(1 for oc in st_open_coding if oc.get("overall_success") == v)
                for v in [0, 1, 2]
            },
            "failure_mode_frequencies": compute_failure_mode_frequencies(st_failure_modes),
            "per_trace": [
                {
                    "test_id": r["test_id"],
                    "scenario": r["scenario"]["tuple"],
                    "mode_correct": r["mode_correct"],
                    "metrics": s,
                    "overall_success": oc.get("overall_success"),
                    "failure_modes": {k: fm.get(k, 0) for k in FAILURE_MODES},
                }
                for r, s, oc, fm in zip(st_results, st_scores, st_open_coding, st_failure_modes)
            ],
        },
        "multi_turn": {
            "total_scenarios": len(mt_results),
            "avg_conversation_completeness": _avg(mt_scores, "Conversation Completeness"),
            "avg_knowledge_retention": _avg(mt_scores, "Knowledge Retention"),
            "avg_role_adherence": _avg(mt_scores, "Role Adherence"),
            "overall_success_distribution": {
                str(v): sum(1 for oc in mt_open_coding if oc.get("overall_success") == v)
                for v in [0, 1, 2]
            },
            "failure_mode_frequencies": compute_failure_mode_frequencies(mt_failure_modes),
            "per_trace": [
                {
                    "test_id": r["test_id"],
                    "scenario": r["scenario"]["tuple"],
                    "num_turns": len(r["turns"]),
                    "metrics": s,
                    "overall_success": oc.get("overall_success"),
                    "failure_modes": {k: fm.get(k, 0) for k in FAILURE_MODES},
                }
                for r, s, oc, fm in zip(mt_results, mt_scores, mt_open_coding, mt_failure_modes)
            ],
        },
    }

    path = ERROR_ANALYSIS_DIR / "baseline_scores.json"
    path.write_text(json.dumps(baseline, indent=2))
    print(f"  Saved → {path}")
    return baseline


def _fmt(val: Optional[float]) -> str:
    return f"{val:.3f}" if val is not None else "n/a"


def _pct(val: Optional[float]) -> str:
    return f"{val:.0%}" if val is not None else "n/a"


def print_report(baseline: Dict, st_results: List[Dict], st_scores: List[Dict],
                 mt_results: List[Dict], mt_scores: List[Dict],
                 st_open_coding: List[Dict], mt_open_coding: List[Dict],
                 tracker: "CostTracker"):
    W = 70
    st = baseline["single_turn"]
    mt = baseline["multi_turn"]

    print("\n" + "=" * W)
    print("  EVALUATION REPORT — JO'S BIKE SHOP CHATBOT")
    print("=" * W)
    print(f"  Chatbot model : {baseline['metadata']['chatbot_model']}")
    print(f"  Judge model   : {baseline['metadata']['judge_model']}")
    print(f"  Timestamp     : {baseline['metadata']['timestamp']}")
    print(f"  Scenarios     : {st['total_scenarios']} single-turn + {mt['total_scenarios']} multi-turn")

    # ── Metric definitions ───────────────────────────────────────────────────
    print("\n" + "-" * W)
    print("  METRICS USED (scores are 0.0 – 1.0; higher is better)")
    print("-" * W)
    definitions = [
        ("Single-Turn", [
            ("Answer Relevancy",
             "Measures whether the bot's response directly addresses the user's query "
             "without introducing irrelevant statements. Evaluated by decomposing the "
             "response into statements and checking each against the input."),
            ("Task Completion",
             "Measures whether the response fully satisfies the user's request — "
             "e.g., providing hours/address for a shop-info query, or actionable tips "
             "for a maintenance query. Evaluated using custom criteria via GEval."),
            ("Conversation Quality",
             "Measures factual accuracy (shop details must match ground truth), "
             "tone (friendly and professional), and appropriate response length. "
             "Evaluated using custom criteria via GEval."),
            ("Mode Detection Accuracy",
             "Deterministic check: did the Burr state machine route to the correct "
             "action (e.g., shop_info, maintenance_tips) for each query?"),
        ]),
        ("Multi-Turn", [
            ("Conversation Completeness",
             "Measures whether the chatbot covered all topics the user raised across "
             "the full conversation, without leaving questions unanswered."),
            ("Knowledge Retention",
             "Measures whether the bot correctly recalls and applies information "
             "provided by the user in earlier turns (e.g., service type, preferred date)."),
            ("Role Adherence",
             "Measures whether the bot stayed in character as a bike shop assistant "
             "and did not drift into off-topic or inappropriate responses."),
        ]),
        ("Both", [
            ("Overall Success (open coding)",
             "LLM-assigned holistic rating: 2 = complete success, "
             "1 = partial success with minor issues, 0 = clear failure."),
            ("Failure Mode Flags (FM1–FM12)",
             "Binary flags per trace indicating whether a specific failure mode "
             "was observed (e.g., FM2 = bot asked for info the user already gave)."),
        ]),
    ]
    for section, items in definitions:
        print(f"\n  [{section}]")
        for name, desc in items:
            print(f"  • {name}")
            for line in _wrap(desc, width=W - 6, indent="    "):
                print(line)

    # ── Single-turn results ──────────────────────────────────────────────────
    print("\n" + "-" * W)
    print("  SINGLE-TURN RESULTS (8 scenarios)")
    print("-" * W)
    print(f"  Mode detection accuracy : {_pct(st['mode_detection_accuracy'])}")
    print(f"  Avg answer relevancy    : {_fmt(st['avg_answer_relevancy'])}")
    print(f"  Avg task completion     : {_fmt(st['avg_task_completion'])}")
    print(f"  Avg conversation quality: {_fmt(st['avg_conversation_quality'])}")
    dist = st["overall_success_distribution"]
    print(f"  Success dist.  complete={dist.get('2',0)}  partial={dist.get('1',0)}  failed={dist.get('0',0)}")

    print(f"\n  {'ID':<7} {'Scenario':<42} {'AR':>5} {'TC':>5} {'CQ':>5} {'OK':>3}")
    print(f"  {'-'*7} {'-'*42} {'-'*5} {'-'*5} {'-'*5} {'-'*3}")
    for r, s, oc in zip(st_results, st_scores, st_open_coding):
        sc = r["scenario"]["tuple"].replace("(", "").replace(")", "")[:42]
        ar = _fmt(s.get("Answer Relevancy", {}).get("score"))
        tc = _fmt(s.get("Task Completion", {}).get("score"))
        cq = _fmt(s.get("Conversation Quality", {}).get("score"))
        ok = str(oc.get("overall_success", "?"))
        print(f"  {r['test_id']:<7} {sc:<42} {ar:>5} {tc:>5} {cq:>5} {ok:>3}")

    # ── Multi-turn results ───────────────────────────────────────────────────
    print("\n" + "-" * W)
    print("  MULTI-TURN RESULTS (16 scenarios)")
    print("-" * W)
    print(f"  Avg conversation completeness: {_fmt(mt['avg_conversation_completeness'])}")
    print(f"  Avg knowledge retention      : {_fmt(mt['avg_knowledge_retention'])}")
    print(f"  Avg role adherence           : {_fmt(mt['avg_role_adherence'])}")
    dist = mt["overall_success_distribution"]
    print(f"  Success dist.  complete={dist.get('2',0)}  partial={dist.get('1',0)}  failed={dist.get('0',0)}")

    print(f"\n  {'ID':<7} {'Scenario':<38} {'Trns':>4} {'CC':>5} {'KR':>5} {'RA':>5} {'OK':>3}")
    print(f"  {'-'*7} {'-'*38} {'-'*4} {'-'*5} {'-'*5} {'-'*5} {'-'*3}")
    for r, s, oc in zip(mt_results, mt_scores, mt_open_coding):
        sc = r["scenario"]["tuple"].replace("(", "").replace(")", "")[:38]
        cc = _fmt(s.get("Conversation Completeness", {}).get("score"))
        kr = _fmt(s.get("Knowledge Retention", {}).get("score"))
        ra = _fmt(s.get("Role Adherence", {}).get("score"))
        ok = str(oc.get("overall_success", "?"))
        trns = str(len(r["turns"]))
        print(f"  {r['test_id']:<7} {sc:<38} {trns:>4} {cc:>5} {kr:>5} {ra:>5} {ok:>3}")

    # ── Failure mode analysis ────────────────────────────────────────────────
    print("\n" + "-" * W)
    print("  FAILURE MODE FREQUENCY (combined single + multi-turn)")
    print("-" * W)
    combined_fm: Dict[str, int] = {}
    for k in FAILURE_MODES:
        combined_fm[k] = (
            st["failure_mode_frequencies"].get(k, 0)
            + mt["failure_mode_frequencies"].get(k, 0)
        )
    total_traces = st["total_scenarios"] + mt["total_scenarios"]
    any_fm = False
    for k, v in sorted(combined_fm.items(), key=lambda x: -x[1]):
        if v > 0:
            any_fm = True
            bar = "█" * v + "░" * (total_traces - v)
            pct = v / total_traces
            print(f"  {k:<40} {v:>2}/{total_traces}  {pct:>5.0%}  {bar}")
            desc = FAILURE_MODES[k]
            for line in _wrap(desc, width=W - 6, indent="    "):
                print(line)
    if not any_fm:
        print("  No failure modes detected.")

    # ── Open coding highlights ───────────────────────────────────────────────
    print("\n" + "-" * W)
    print("  NOTABLE ISSUES (from open coding)")
    print("-" * W)
    issues_found = False
    for label, results, codings in [
        ("ST", st_results, st_open_coding),
        ("MT", mt_results, mt_open_coding),
    ]:
        for r, oc in zip(results, codings):
            wrong = oc.get("what_went_wrong", "None")
            notable = oc.get("notable_behaviors", "None")
            if wrong.lower() not in ("none", "", "n/a") or notable.lower() not in ("none", "", "n/a"):
                issues_found = True
                print(f"\n  {r['test_id']} — {r['scenario']['tuple']}")
                if wrong.lower() not in ("none", "", "n/a"):
                    for line in _wrap(f"Issue: {wrong}", width=W - 4, indent="    "):
                        print(line)
                if notable.lower() not in ("none", "", "n/a"):
                    for line in _wrap(f"Notable: {notable}", width=W - 4, indent="    "):
                        print(line)
    if not issues_found:
        print("  No notable issues found.")

    # ── Cost estimate ────────────────────────────────────────────────────────
    print("\n" + "-" * W)
    print("  ESTIMATED COST")
    print("-" * W)
    cost = baseline["metadata"]["cost"]
    bot = cost["bot"]
    judge = cost["judge"]
    print(f"  Bot ({bot['model']}) — estimated (4 chars ≈ 1 token):")
    print(f"    Input tokens  : ~{bot['input_tokens_estimated']:,}")
    print(f"    Output tokens : ~{bot['output_tokens_estimated']:,}")
    print(f"    Cost          : ~${bot['cost_usd_estimated']:.4f} USD")
    print(f"  Judge ({judge['model']}):")
    print(f"    DeepEval metrics : ${judge['deepeval_metrics_cost_usd']:.4f} USD")
    print(f"    Direct API calls : ${judge['direct_calls_cost_usd']:.4f} USD  "
          f"({judge['direct_calls_input_tokens']:,} in / {judge['direct_calls_output_tokens']:,} out tokens)")
    print(f"    Total judge      : ${judge['total_judge_cost_usd']:.4f} USD")
    print(f"  ──────────────────────────────────────────────")
    print(f"  TOTAL ESTIMATED COST : ${cost['total_cost_usd']:.4f} USD")
    print(f"  Note: {cost['note']}")

    print("\n" + "=" * W)
    print(f"  Full results saved to: {ERROR_ANALYSIS_DIR}/")
    print(f"  Baseline snapshot   : {ERROR_ANALYSIS_DIR}/baseline_scores.json")
    print("=" * W)


def save_markdown_report(
    baseline: Dict,
    st_results: List[Dict], st_scores: List[Dict],
    mt_results: List[Dict], mt_scores: List[Dict],
    st_open_coding: List[Dict], mt_open_coding: List[Dict],
    st_failure_modes: List[Dict], mt_failure_modes: List[Dict],
    tracker: "CostTracker",
) -> Path:
    """Generate a detailed markdown evaluation report and save it to error_analysis/."""
    sydney_tz = ZoneInfo("Australia/Sydney")
    now_sydney = datetime.now(sydney_tz)
    filename = f"eval_report_{now_sydney.strftime('%Y%m%d_%H%M%S')}_AEST.md"
    path = ERROR_ANALYSIS_DIR / filename

    st = baseline["single_turn"]
    mt = baseline["multi_turn"]
    meta = baseline["metadata"]

    lines: List[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"\n{'#' * level} {text}\n")

    def p(text: str = "") -> None:
        lines.append(text)

    def table_row(*cols: str) -> str:
        return "| " + " | ".join(str(c) for c in cols) + " |"

    def table_sep(*widths: int) -> str:
        return "| " + " | ".join("-" * w for w in widths) + " |"

    def score_badge(val: Optional[float]) -> str:
        if val is None:
            return "n/a"
        if val >= 0.8:
            return f"🟢 {val:.3f}"
        if val >= 0.5:
            return f"🟡 {val:.3f}"
        return f"🔴 {val:.3f}"

    def ok_badge(val: Any) -> str:
        if val == 2:
            return "✅ Pass"
        if val == 1:
            return "⚠️ Partial"
        return "❌ Fail"

    # ── Title ──────────────────────────────────────────────────────────────────
    lines.append(f"# JO's Bike Shop Chatbot — Evaluation Report")
    p(f"> **Run date (Sydney):** {now_sydney.strftime('%A, %d %B %Y %H:%M:%S %Z')}  ")
    p(f"> **Chatbot model:** `{meta['chatbot_model']}`  ")
    p(f"> **Judge model:** `{meta['judge_model']}`  ")
    p(f"> **Scenarios:** {st['total_scenarios']} single-turn + {mt['total_scenarios']} multi-turn  ")
    p(f"> **Baseline snapshot:** `{ERROR_ANALYSIS_DIR}/baseline_scores.json`")

    # ── Executive Summary ──────────────────────────────────────────────────────
    h(2, "Executive Summary")

    p("### Single-Turn Performance\n")
    p(f"| Metric | Score |")
    p(table_sep(35, 15))
    p(table_row("Mode Detection Accuracy", f"{st['mode_detection_accuracy']:.0%}" if st['mode_detection_accuracy'] is not None else "n/a"))
    p(table_row("Avg Answer Relevancy", score_badge(st['avg_answer_relevancy'])))
    p(table_row("Avg Task Completion", score_badge(st['avg_task_completion'])))
    p(table_row("Avg Conversation Quality", score_badge(st['avg_conversation_quality'])))
    dist = st["overall_success_distribution"]
    p(table_row("Pass / Partial / Fail", f"{dist.get('2',0)} / {dist.get('1',0)} / {dist.get('0',0)}"))

    p("\n### Multi-Turn Performance\n")
    p(f"| Metric | Score |")
    p(table_sep(35, 15))
    p(table_row("Avg Conversation Completeness", score_badge(mt['avg_conversation_completeness'])))
    p(table_row("Avg Knowledge Retention", score_badge(mt['avg_knowledge_retention'])))
    p(table_row("Avg Role Adherence", score_badge(mt['avg_role_adherence'])))
    dist = mt["overall_success_distribution"]
    p(table_row("Pass / Partial / Fail", f"{dist.get('2',0)} / {dist.get('1',0)} / {dist.get('0',0)}"))

    # top failure modes
    combined_fm: Dict[str, int] = {
        k: st["failure_mode_frequencies"].get(k, 0) + mt["failure_mode_frequencies"].get(k, 0)
        for k in FAILURE_MODES
    }
    top_fm = [(k, v) for k, v in sorted(combined_fm.items(), key=lambda x: -x[1]) if v > 0]

    p("\n### Top Failure Modes\n")
    if top_fm:
        p("| Failure Mode | Occurrences | Rate |")
        p(table_sep(42, 13, 8))
        total_traces = st["total_scenarios"] + mt["total_scenarios"]
        for k, v in top_fm[:6]:
            p(table_row(k, f"{v}/{total_traces}", f"{v/total_traces:.0%}"))
    else:
        p("_No failure modes detected._")

    # ── Cost Estimate ──────────────────────────────────────────────────────────
    h(2, "Cost Estimate")
    cost = baseline["metadata"]["cost"]
    bot_c = cost["bot"]
    judge_c = cost["judge"]

    p("| Component | Detail | Value |")
    p(table_sep(30, 42, 18))
    p(table_row(f"Bot ({bot_c['model']})", "Input tokens (estimated)", f"~{bot_c['input_tokens_estimated']:,}"))
    p(table_row("", "Output tokens (estimated)", f"~{bot_c['output_tokens_estimated']:,}"))
    p(table_row("", "**Bot cost (estimated)**", f"**~${bot_c['cost_usd_estimated']:.4f} USD**"))
    p(table_row(f"Judge ({judge_c['model']})", "DeepEval metrics cost", f"${judge_c['deepeval_metrics_cost_usd']:.4f} USD"))
    p(table_row("", "Direct API calls — input tokens", f"{judge_c['direct_calls_input_tokens']:,}"))
    p(table_row("", "Direct API calls — output tokens", f"{judge_c['direct_calls_output_tokens']:,}"))
    p(table_row("", "Direct API calls cost", f"${judge_c['direct_calls_cost_usd']:.4f} USD"))
    p(table_row("", "**Total judge cost**", f"**${judge_c['total_judge_cost_usd']:.4f} USD**"))
    p(table_row("**TOTAL**", "", f"**${cost['total_cost_usd']:.4f} USD**"))
    p(f"\n> _{cost['note']}_\n")

    # ── Metric Definitions ─────────────────────────────────────────────────────
    h(2, "Metric Definitions")
    p("All DeepEval scores are in the range **0.0 – 1.0** (higher is better).")
    p("🟢 ≥ 0.8 &nbsp; 🟡 0.5–0.8 &nbsp; 🔴 < 0.5\n")

    h(3, "Single-Turn Metrics")
    metric_defs_st = [
        ("Answer Relevancy", "AnswerRelevancyMetric",
         "Decomposes the bot's response into statements and checks each against the user's input. "
         "Penalises responses that introduce irrelevant information or ignore part of the query."),
        ("Task Completion", "GEval (custom criteria)",
         "Evaluates whether the response fully satisfies the user's intent using mode-specific criteria — "
         "e.g., shop-info must include hours/address/contact; maintenance tips must include actionable advice."),
        ("Conversation Quality", "GEval (custom criteria)",
         "Checks three dimensions: (1) factual accuracy against the shop's ground-truth data, "
         "(2) tone — friendly and professional, (3) length — not too verbose or too brief."),
        ("Mode Detection Accuracy", "Deterministic",
         "Compares the Burr state machine's routed action (e.g. `shop_info`, `maintenance_tips`) "
         "against the expected mode for each scenario. No LLM involved."),
    ]
    for name, tool, desc in metric_defs_st:
        p(f"**{name}** _(via {tool})_  ")
        p(f"{desc}\n")

    h(3, "Multi-Turn Metrics")
    metric_defs_mt = [
        ("Conversation Completeness", "ConversationCompletenessMetric",
         "Measures whether the chatbot addressed all topics and questions the user raised across the "
         "full conversation, without leaving anything unanswered."),
        ("Knowledge Retention", "KnowledgeRetentionMetric",
         "Measures whether the bot correctly recalls and uses information the user provided in earlier "
         "turns — e.g., not re-asking for the service type after the user already stated it."),
        ("Role Adherence", "RoleAdherenceMetric",
         "Measures whether the bot stayed in character as a JO's Bike Shop assistant throughout the "
         "conversation and did not drift into off-topic or inappropriate territory."),
    ]
    for name, tool, desc in metric_defs_mt:
        p(f"**{name}** _(via {tool})_  ")
        p(f"{desc}\n")

    h(3, "Qualitative Metrics (both turn types)")
    p("**Overall Success** _(LLM open coding)_  ")
    p("Holistic rating assigned by the judge LLM after reviewing the full trace: "
      "`2` = complete success, `1` = partial success with minor issues, `0` = clear failure.\n")
    p("**Failure Mode Flags FM1–FM12** _(LLM classification)_  ")
    p("Binary flags (0/1) per trace indicating whether a specific failure pattern was observed. "
      "See the Failure Mode Analysis section for full definitions.\n")

    # ── Single-Turn Detailed Results ───────────────────────────────────────────
    h(2, "Single-Turn Detailed Results")
    p(f"_{len(st_results)} scenarios covering shop_info, maintenance_tips, policy_question, and what_can_you_do intents._\n")

    p("| ID | Scenario Tuple | AR | TC | CQ | Result |")
    p(table_sep(6, 48, 7, 7, 7, 10))
    for r, s, oc in zip(st_results, st_scores, st_open_coding):
        sc = r["scenario"]["tuple"]
        ar = _fmt(s.get("Answer Relevancy", {}).get("score"))
        tc = _fmt(s.get("Task Completion", {}).get("score"))
        cq = _fmt(s.get("Conversation Quality", {}).get("score"))
        ok = ok_badge(oc.get("overall_success"))
        p(table_row(r["test_id"], sc, ar, tc, cq, ok))

    p("\n_AR = Answer Relevancy · TC = Task Completion · CQ = Conversation Quality_\n")

    for r, s, oc in zip(st_results, st_scores, st_open_coding):
        h(3, f"{r['test_id']} — {r['scenario']['tuple']}")
        p(f"**User query:** {r['user_query']}  ")
        p(f"**Expected mode:** `{r['expected_mode']}` · **Action taken:** `{r['action_taken']}` · "
          f"**Mode correct:** {'Yes' if r['mode_correct'] else 'No'}\n")
        p(f"**Bot response:**")
        p(f"> {r['bot_response'].replace(chr(10), '  \n> ')}\n")
        p(f"| Metric | Score | Reason |")
        p(table_sep(24, 7, 50))
        for metric_name in ["Answer Relevancy", "Task Completion", "Conversation Quality"]:
            ms = s.get(metric_name, {})
            reason = (ms.get("reason") or "").replace("|", "\\|")[:120]
            p(table_row(metric_name, _fmt(ms.get("score")), reason))
        p(f"\n**Open coding**  ")
        p(f"- What worked: {oc.get('what_worked', 'n/a')}  ")
        p(f"- What went wrong: {oc.get('what_went_wrong', 'n/a')}  ")
        p(f"- Notable: {oc.get('notable_behaviors', 'n/a')}  ")
        p(f"- **Overall success: {ok_badge(oc.get('overall_success'))}**\n")

    # ── Multi-Turn Detailed Results ────────────────────────────────────────────
    h(2, "Multi-Turn Detailed Results")
    p(f"_{len(mt_results)} scenarios covering all 7 intents with varied completeness, "
      f"interaction patterns, and user behaviours._\n")

    p("| ID | Scenario Tuple | Turns | CC | KR | RA | Result |")
    p(table_sep(6, 46, 5, 7, 7, 7, 10))
    for r, s, oc in zip(mt_results, mt_scores, mt_open_coding):
        sc = r["scenario"]["tuple"]
        cc = _fmt(s.get("Conversation Completeness", {}).get("score"))
        kr = _fmt(s.get("Knowledge Retention", {}).get("score"))
        ra = _fmt(s.get("Role Adherence", {}).get("score"))
        ok = ok_badge(oc.get("overall_success"))
        p(table_row(r["test_id"], sc, len(r["turns"]), cc, kr, ra, ok))

    p("\n_CC = Conversation Completeness · KR = Knowledge Retention · RA = Role Adherence_\n")

    for r, s, oc in zip(mt_results, mt_scores, mt_open_coding):
        h(3, f"{r['test_id']} — {r['scenario']['tuple']}")
        p(f"**User goal:** {r['user_goal']}  ")
        p(f"**Turns:** {len(r['turns'])} · **Scenario:** `{r['scenario']['interaction_pattern']}` / "
          f"`{r['scenario']['user_behavior']}` / `{r['scenario']['completeness']}`\n")
        p("**Conversation transcript:**\n")
        for turn in r["turns"]:
            role_label = "**User**" if turn["role"] == "user" else "**Bot**"
            p(f"{role_label}: {turn['content']}  ")
        p(f"\n| Metric | Score | Reason |")
        p(table_sep(28, 7, 50))
        for metric_name in ["Conversation Completeness", "Knowledge Retention", "Role Adherence"]:
            ms = s.get(metric_name, {})
            reason = (ms.get("reason") or "").replace("|", "\\|")[:120]
            p(table_row(metric_name, _fmt(ms.get("score")), reason))
        p(f"\n**Open coding**  ")
        p(f"- What worked: {oc.get('what_worked', 'n/a')}  ")
        p(f"- What went wrong: {oc.get('what_went_wrong', 'n/a')}  ")
        p(f"- Notable: {oc.get('notable_behaviors', 'n/a')}  ")
        p(f"- **Overall success: {ok_badge(oc.get('overall_success'))}**\n")

    # ── Failure Mode Analysis ──────────────────────────────────────────────────
    h(2, "Failure Mode Analysis")
    p("Each trace is evaluated against 12 pre-defined failure modes. "
      "A flag of `1` means the failure was observed in that trace.\n")

    h(3, "Failure Mode Definitions")
    for k, desc in FAILURE_MODES.items():
        p(f"**{k}:** {desc}  ")

    h(3, "Frequency Summary")
    total_traces = st["total_scenarios"] + mt["total_scenarios"]
    p(f"| Failure Mode | Single-Turn | Multi-Turn | Total | Rate |")
    p(table_sep(42, 12, 11, 7, 6))
    for k in FAILURE_MODES:
        st_v = st["failure_mode_frequencies"].get(k, 0)
        mt_v = mt["failure_mode_frequencies"].get(k, 0)
        total = st_v + mt_v
        rate = f"{total/total_traces:.0%}" if total > 0 else "0%"
        p(table_row(k, st_v, mt_v, total, rate))

    h(3, "Per-Trace Failure Mode Flags")
    fm_keys = list(FAILURE_MODES.keys())
    # Split into two tables for readability
    half = len(fm_keys) // 2
    for chunk in [fm_keys[:half], fm_keys[half:]]:
        header = ["ID"] + [k.split("_")[0] for k in chunk]
        p("| " + " | ".join(header) + " |")
        p("| " + " | ".join(["-" * max(4, len(h)) for h in header]) + " |")
        for r, fm in zip(st_results, st_failure_modes):
            row = [r["test_id"]] + [str(fm.get(k, 0)) for k in chunk]
            p("| " + " | ".join(row) + " |")
        for r, fm in zip(mt_results, mt_failure_modes):
            row = [r["test_id"]] + [str(fm.get(k, 0)) for k in chunk]
            p("| " + " | ".join(row) + " |")
        p()

    # ── Footer ─────────────────────────────────────────────────────────────────
    h(2, "Output Files")
    p("| File | Description |")
    p(table_sep(45, 50))
    p(table_row("`error_analysis/baseline_scores.json`", "Machine-readable metric scores — use for before/after model comparison"))
    p(table_row("`error_analysis/single_turn_results.json`", "Raw single-turn bot responses"))
    p(table_row("`error_analysis/multi_turn_results.json`", "Raw multi-turn conversation traces"))
    p(table_row("`error_analysis/single_turn_analysis.csv`", "Per-trace scores + open coding + failure mode flags"))
    p(table_row("`error_analysis/multi_turn_analysis.csv`", "Per-trace scores + open coding + failure mode flags"))
    p(table_row(f"`error_analysis/{filename}`", "This report"))
    p(f"\n---\n_Generated by `eval_runner.py` · {now_sydney.strftime('%d %b %Y %H:%M %Z')}_")

    path.write_text("\n".join(lines))
    return path


def _wrap(text: str, width: int, indent: str = "") -> List[str]:
    """Simple word-wrap returning a list of indented lines."""
    words = text.split()
    lines = []
    current = indent
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = indent + word
        else:
            current = current + (" " if current != indent else "") + word
    if current.strip():
        lines.append(current)
    return lines


# ─── Main ─────────────────────────────────────────────────────────────────────


async def main(judge_model: str, start_step: int = 1):
    timestamp = datetime.now(timezone.utc).isoformat()
    client = OpenAI()
    tracker = CostTracker(bot_model="gpt-4o", judge_model=judge_model)

    start_step = _resolve_start_step(start_step)

    print("\n" + "=" * 60)
    print("  JO'S BIKE SHOP — CHATBOT EVALUATION RUNNER")
    print("=" * 60)
    print(f"  This script evaluates the JO's Bike Shop chatbot across")
    print(f"  24 test scenarios (8 single-turn + 16 multi-turn) using")
    print(f"  DeepEval metrics and LLM-assisted qualitative analysis.")
    print(f"\n  Judge model   : {judge_model}  (scores the responses)")
    print(f"  Chatbot model : gpt-4o  (powers the chatbot being tested)")
    print(f"  Starting step : {start_step} of 6")
    print(f"  Timestamp     : {timestamp}")
    print(f"\n  Results will be saved to: {ERROR_ANALYSIS_DIR}/")
    print(f"  Step cache stored in    : {CACHE_DIR}/")
    print("=" * 60)

    # Load synthetic examples (always needed)
    st_examples = json.loads(
        (ERROR_ANALYSIS_DIR / "single_turn_synthetic_examples.json").read_text()
    )["examples"]
    mt_examples = json.loads(
        (ERROR_ANALYSIS_DIR / "multi_turn_synthetic_examples.json").read_text()
    )["examples"]

    # ── Step 1: Run single-turn bot ───────────────────────────────────────────
    if start_step <= 1:
        print(f"\n{'='*60}")
        print(f"  STEP 1/6 — Run single-turn scenarios")
        print(f"{'='*60}")
        print(f"  Each of the {len(st_examples)} single-turn queries is sent directly")
        print(f"  to the Burr chatbot. We record the bot's response and which")
        print(f"  action (mode) the state machine routed to.")
        print()
        st_results = await run_single_turn_scenarios(st_examples, tracker)
        _save_cache("st_results", [{k: v for k, v in r.items()} for r in st_results])
        print(f"\n  Done. {len(st_results)} responses collected. Cache saved.")
    else:
        print(f"\n  [cached] STEP 1 — {len(st_examples)} single-turn results loaded from cache.")
        st_results = _load_cache("st_results")

    # ── Step 2: Score single-turn ─────────────────────────────────────────────
    if start_step <= 2:
        print(f"\n{'='*60}")
        print(f"  STEP 2/6 — Score single-turn traces with DeepEval")
        print(f"{'='*60}")
        print(f"  Running 3 metrics per trace ({len(st_results)} traces × 3 = {len(st_results)*3} LLM calls):")
        print(f"    • Answer Relevancy   — did the response address the question?")
        print(f"    • Task Completion    — was the user's request fully satisfied?")
        print(f"    • Conversation Quality — factually accurate, right tone & length?")
        print(f"  Metrics run in parallel per trace via asyncio.gather().")
        print()
        st_test_cases = build_single_turn_test_cases(st_results)
        st_scores = await run_single_turn_metrics(st_test_cases, judge_model, tracker)
        _save_cache("st_scores", st_scores)
        print(f"\n  Done. {len(st_scores)} score sets collected. Cache saved.")
    else:
        print(f"  [cached] STEP 2 — single-turn scores loaded from cache.")
        st_scores = _load_cache("st_scores")

    # ── Step 3: Simulate multi-turn ───────────────────────────────────────────
    if start_step <= 3:
        print(f"\n{'='*60}")
        print(f"  STEP 3/6 — Simulate multi-turn conversations")
        print(f"{'='*60}")
        print(f"  DeepEval's ConversationSimulator plays the user role for each of")
        print(f"  the {len(mt_examples)} multi-turn scenarios. It generates realistic user")
        print(f"  messages turn-by-turn, calling the live Burr chatbot for each")
        print(f"  bot response. Each scenario runs up to 8 user-bot exchanges.")
        print(f"  Simulations run sequentially (one Burr app instance per scenario).")
        print()
        mt_results = await run_multi_turn_scenarios(mt_examples, judge_model, tracker)
        _save_cache("mt_results", [
            {k: v for k, v in r.items() if k != "test_case"} for r in mt_results
        ])
        total_turns = sum(len(r["turns"]) for r in mt_results)
        print(f"\n  Done. {len(mt_results)} conversations simulated ({total_turns} total turns). Cache saved.")
    else:
        print(f"  [cached] STEP 3 — {len(mt_examples)} multi-turn traces loaded from cache.")
        mt_results = _load_cache("mt_results")
        _rebuild_mt_test_cases(mt_results)

    # ── Step 4: Score multi-turn ──────────────────────────────────────────────
    if start_step <= 4:
        print(f"\n{'='*60}")
        print(f"  STEP 4/6 — Score multi-turn traces with DeepEval")
        print(f"{'='*60}")
        print(f"  Running 3 conversational metrics per trace ({len(mt_results)} traces × 3 = {len(mt_results)*3} LLM calls):")
        print(f"    • Conversation Completeness — did the bot address everything the user raised?")
        print(f"    • Knowledge Retention       — did the bot remember facts from earlier turns?")
        print(f"    • Role Adherence            — did the bot stay in character as a shop assistant?")
        print(f"  Metrics run in parallel per trace via asyncio.gather().")
        print()
        mt_scores = await run_multi_turn_metrics(mt_results, judge_model, tracker)
        _save_cache("mt_scores", mt_scores)
        print(f"\n  Done. {len(mt_scores)} score sets collected. Cache saved.")
    else:
        print(f"  [cached] STEP 4 — multi-turn scores loaded from cache.")
        mt_scores = _load_cache("mt_scores")

    # ── Step 5: Open coding + failure modes ───────────────────────────────────
    if start_step <= 5:
        print(f"\n{'='*60}")
        print(f"  STEP 5/6 — LLM-assisted open coding & failure mode analysis")
        print(f"{'='*60}")
        print(f"  Part A — Open coding ({len(st_results) + len(mt_results)} traces):")
        print(f"    {judge_model} reviews each conversation and identifies what worked,")
        print(f"    what went wrong, and any notable chatbot behaviour patterns.")
        print(f"    Each trace also gets an overall success rating (0=fail, 1=partial, 2=pass).")
        print()
        print(f"  Open coding {len(st_results)} single-turn traces...")
        st_open_coding = run_open_coding(st_results, is_multi_turn=False, client=client, judge_model=judge_model, tracker=tracker)
        print(f"  Open coding {len(mt_results)} multi-turn traces...")
        mt_open_coding = run_open_coding(mt_results, is_multi_turn=True, client=client, judge_model=judge_model, tracker=tracker)

        print(f"\n  Part B — Failure mode classification ({len(st_results) + len(mt_results)} traces × 12 modes):")
        print(f"    Each trace is checked against 12 pre-defined failure modes (FM1–FM12),")
        print(f"    ranging from intent detection errors to cancellation-not-detected.")
        print(f"    Each flag is 1 (failure observed) or 0 (not observed).")
        print()
        print(f"  Classifying single-turn failure modes...")
        st_failure_modes = run_failure_mode_analysis(st_results, st_open_coding, False, client, judge_model, tracker)
        print(f"  Classifying multi-turn failure modes...")
        mt_failure_modes = run_failure_mode_analysis(mt_results, mt_open_coding, True, client, judge_model, tracker)

        _save_cache("st_open_coding", st_open_coding)
        _save_cache("mt_open_coding", mt_open_coding)
        _save_cache("st_failure_modes", st_failure_modes)
        _save_cache("mt_failure_modes", mt_failure_modes)
        print(f"\n  Done. Open coding and failure modes complete. Cache saved.")
    else:
        print(f"  [cached] STEP 5 — open coding and failure modes loaded from cache.")
        st_open_coding = _load_cache("st_open_coding")
        mt_open_coding = _load_cache("mt_open_coding")
        st_failure_modes = _load_cache("st_failure_modes")
        mt_failure_modes = _load_cache("mt_failure_modes")

    # ── Step 6: Save results ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STEP 6/6 — Save all results")
    print(f"{'='*60}")
    print(f"  Writing final output files to {ERROR_ANALYSIS_DIR}/:")
    save_single_turn_results(st_results, st_scores, timestamp)
    save_multi_turn_results(mt_results, timestamp)
    save_single_turn_csv(st_results, st_scores, st_open_coding, st_failure_modes)
    save_multi_turn_csv(mt_results, mt_scores, mt_open_coding, mt_failure_modes)
    baseline = save_baseline_scores(
        st_results, st_scores, st_open_coding, st_failure_modes,
        mt_results, mt_scores, mt_open_coding, mt_failure_modes,
        judge_model=judge_model,
        timestamp=timestamp,
        tracker=tracker,
    )
    report_path = save_markdown_report(
        baseline,
        st_results, st_scores,
        mt_results, mt_scores,
        st_open_coding, mt_open_coding,
        st_failure_modes, mt_failure_modes,
        tracker=tracker,
    )
    print(f"  Saved → {report_path}")
    print(f"\n  All files saved. Generating terminal summary...")

    print_report(
        baseline,
        st_results, st_scores,
        mt_results, mt_scores,
        st_open_coding, mt_open_coding,
        tracker=tracker,
    )

    # Allow pending httpx cleanup coroutines to finish before the loop closes
    await asyncio.sleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all chatbot evaluations")
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"OpenAI model used as evaluator judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help="Resume from this step using cached results from previous run (default: 1)",
    )
    args = parser.parse_args()
    asyncio.run(main(judge_model=args.judge_model, start_step=args.start_step))
