"""
compare_baselines.py — Compare two baseline_scores.json files.

Prints a delta table to the terminal and saves a markdown comparison report
to error_analysis/ with a Sydney-timezone timestamp in the filename.

Usage:
    python compare_baselines.py <old_baseline.json> <new_baseline.json>
    python compare_baselines.py --old baseline_gpt4o.json --new baseline_gemma4.json

The "old" baseline is the reference (e.g. the original GPT-4o run).
The "new" baseline is the one being compared against it (e.g. after swapping the LLM).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

ERROR_ANALYSIS_DIR = Path("error_analysis")

FAILURE_MODES = {
    "FM1_intent_detection_error": "Bot misclassified the user's intent and routed to the wrong handler",
    "FM2_info_extraction_issue": "Bot failed to extract information the user clearly provided",
    "FM3_state_management_problem": "Appointment data or booking state was lost or corrupted across turns",
    "FM4_context_loss": "Bot failed to retain or reference information from earlier turns",
    "FM5_correction_handling_failure": "Bot did not handle a user correction gracefully",
    "FM6_confirmation_flow_issue": "Confirmation step was skipped, repeated, or handled incorrectly",
    "FM7_off_topic_redirection_failure": "Bot did not redirect a distracted user back to the active booking flow",
    "FM8_cancellation_not_detected": "Bot failed to recognise that the user wanted to cancel",
    "FM9_recall_failure": "Bot could not recall or correctly reference a booking made earlier",
    "FM10_response_quality_issue": "Response was factually wrong, incoherent, too brief, or unhelpful",
    "FM11_redundant_questioning": "Bot asked for information the user had already provided",
    "FM12_flow_transition_error": "Bot transitioned between states incorrectly",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _load(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        print(f"Error: file not found: {p}", file=sys.stderr)
        sys.exit(1)
    return json.loads(p.read_text())


def _delta_str(old: Optional[float], new: Optional[float], higher_is_better: bool = True) -> str:
    """Return a formatted delta string with direction arrow."""
    if old is None or new is None:
        return "n/a"
    d = new - old
    if abs(d) < 0.0001:
        return f"   0.000  ="
    arrow = ("↑" if d > 0 else "↓")
    better = (d > 0) == higher_is_better
    sign = "+" if d > 0 else ""
    marker = "✓" if better else "✗"
    return f"{sign}{d:.3f}  {arrow} {marker}"


def _fmt(val: Optional[float]) -> str:
    return f"{val:.3f}" if val is not None else " n/a"


def _pct(val: Optional[float]) -> str:
    return f"{val:.0%}" if val is not None else "n/a"


def _fm_delta(old_v: int, new_v: int) -> str:
    """For failure modes: fewer is better."""
    d = new_v - old_v
    if d == 0:
        return "   0  ="
    sign = "+" if d > 0 else ""
    worse = d > 0
    return f"{sign}{d}  {'↑ worse' if worse else '↓ better'}"


def _score_badge(val: Optional[float]) -> str:
    if val is None:
        return "n/a"
    if val >= 0.8:
        return f"🟢 {val:.3f}"
    if val >= 0.5:
        return f"🟡 {val:.3f}"
    return f"🔴 {val:.3f}"


def _delta_badge(old: Optional[float], new: Optional[float], higher_is_better: bool = True) -> str:
    if old is None or new is None:
        return "n/a"
    d = new - old
    if abs(d) < 0.0001:
        return "➡ 0.000"
    better = (d > 0) == higher_is_better
    sign = "+" if d > 0 else ""
    icon = "🟢" if better else "🔴"
    return f"{icon} {sign}{d:.3f}"


def _wrap(text: str, width: int, indent: str = "") -> List[str]:
    words = text.split()
    lines_out = []
    current = indent
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines_out.append(current)
            current = indent + word
        else:
            current = current + (" " if current != indent else "") + word
    if current.strip():
        lines_out.append(current)
    return lines_out


# ─── CLI report ───────────────────────────────────────────────────────────────


def print_comparison(old: Dict, new: Dict) -> None:
    W = 72
    old_meta = old["metadata"]
    new_meta = new["metadata"]
    old_st = old["single_turn"]
    new_st = new["single_turn"]
    old_mt = old["multi_turn"]
    new_mt = new["multi_turn"]

    print("\n" + "=" * W)
    print("  BASELINE COMPARISON — JO'S BIKE SHOP CHATBOT")
    print("=" * W)
    print(f"  Old run  : {old_meta.get('timestamp', 'unknown')}")
    print(f"    Chatbot: {old_meta.get('chatbot_model', '?')}   Judge: {old_meta.get('judge_model', '?')}")
    print(f"  New run  : {new_meta.get('timestamp', 'unknown')}")
    print(f"    Chatbot: {new_meta.get('chatbot_model', '?')}   Judge: {new_meta.get('judge_model', '?')}")

    if old_meta.get("judge_model") != new_meta.get("judge_model"):
        print(f"\n  ⚠  WARNING: Judge models differ — score changes may reflect")
        print(f"     judge calibration differences, not chatbot quality changes.")
        print(f"     Recommendation: keep the judge model fixed when swapping chatbots.")

    # ── Single-turn ──────────────────────────────────────────────────────────
    print(f"\n{'-' * W}")
    print("  SINGLE-TURN METRICS")
    print(f"{'-' * W}")
    col_w = 32
    print(f"  {'Metric':<{col_w}} {'Old':>7}  {'New':>7}  {'Delta':<18}")
    print(f"  {'-'*col_w} {'-'*7}  {'-'*7}  {'-'*18}")

    st_metrics = [
        ("Mode Detection Accuracy", "mode_detection_accuracy", True),
        ("Avg Answer Relevancy",    "avg_answer_relevancy",    True),
        ("Avg Task Completion",     "avg_task_completion",     True),
        ("Avg Conversation Quality","avg_conversation_quality",True),
    ]
    for label, key, hib in st_metrics:
        o = old_st.get(key)
        n = new_st.get(key)
        print(f"  {label:<{col_w}} {_fmt(o):>7}  {_fmt(n):>7}  {_delta_str(o, n, hib)}")

    o_dist = old_st.get("overall_success_distribution", {})
    n_dist = new_st.get("overall_success_distribution", {})
    print(f"\n  Overall success distribution (pass / partial / fail):")
    print(f"    Old: {o_dist.get('2',0)} / {o_dist.get('1',0)} / {o_dist.get('0',0)}")
    print(f"    New: {n_dist.get('2',0)} / {n_dist.get('1',0)} / {n_dist.get('0',0)}")

    # ── Multi-turn ───────────────────────────────────────────────────────────
    print(f"\n{'-' * W}")
    print("  MULTI-TURN METRICS")
    print(f"{'-' * W}")
    print(f"  {'Metric':<{col_w}} {'Old':>7}  {'New':>7}  {'Delta':<18}")
    print(f"  {'-'*col_w} {'-'*7}  {'-'*7}  {'-'*18}")

    mt_metrics = [
        ("Avg Conversation Completeness", "avg_conversation_completeness", True),
        ("Avg Knowledge Retention",       "avg_knowledge_retention",       True),
        ("Avg Role Adherence",            "avg_role_adherence",            True),
    ]
    for label, key, hib in mt_metrics:
        o = old_mt.get(key)
        n = new_mt.get(key)
        print(f"  {label:<{col_w}} {_fmt(o):>7}  {_fmt(n):>7}  {_delta_str(o, n, hib)}")

    o_dist = old_mt.get("overall_success_distribution", {})
    n_dist = new_mt.get("overall_success_distribution", {})
    print(f"\n  Overall success distribution (pass / partial / fail):")
    print(f"    Old: {o_dist.get('2',0)} / {o_dist.get('1',0)} / {o_dist.get('0',0)}")
    print(f"    New: {n_dist.get('2',0)} / {n_dist.get('1',0)} / {n_dist.get('0',0)}")

    # ── Failure modes ────────────────────────────────────────────────────────
    print(f"\n{'-' * W}")
    print("  FAILURE MODE CHANGES  (old → new, fewer occurrences = better)")
    print(f"{'-' * W}")
    total_old = old_st["total_scenarios"] + old_mt["total_scenarios"]
    total_new = new_st["total_scenarios"] + new_mt["total_scenarios"]

    old_fm_combined = {
        k: old_st["failure_mode_frequencies"].get(k, 0) + old_mt["failure_mode_frequencies"].get(k, 0)
        for k in FAILURE_MODES
    }
    new_fm_combined = {
        k: new_st["failure_mode_frequencies"].get(k, 0) + new_mt["failure_mode_frequencies"].get(k, 0)
        for k in FAILURE_MODES
    }

    # Sort by absolute change descending
    fm_rows = sorted(
        [(k, old_fm_combined[k], new_fm_combined[k]) for k in FAILURE_MODES],
        key=lambda x: abs(x[2] - x[1]),
        reverse=True,
    )
    any_change = any(o != n for _, o, n in fm_rows)
    if any_change:
        print(f"  {'Failure Mode':<42} {'Old':>5}  {'New':>5}  {'Change'}")
        print(f"  {'-'*42} {'-'*5}  {'-'*5}  {'-'*20}")
        for k, o, n in fm_rows:
            if o == 0 and n == 0:
                continue
            rate_old = f"{o}/{total_old}"
            rate_new = f"{n}/{total_new}"
            print(f"  {k:<42} {rate_old:>5}  {rate_new:>5}  {_fm_delta(o, n)}")
    else:
        print("  No change in failure mode frequencies.")

    # ── Cost comparison ──────────────────────────────────────────────────────
    old_cost = old_meta.get("cost")
    new_cost = new_meta.get("cost")
    if old_cost and new_cost:
        print(f"\n{'-' * W}")
        print("  COST COMPARISON")
        print(f"{'-' * W}")
        o_bot = old_cost["bot"]["cost_usd_estimated"]
        n_bot = new_cost["bot"]["cost_usd_estimated"]
        o_judge = old_cost["judge"]["total_judge_cost_usd"]
        n_judge = new_cost["judge"]["total_judge_cost_usd"]
        o_total = old_cost["total_cost_usd"]
        n_total = new_cost["total_cost_usd"]
        col_w = 28
        print(f"  {'Component':<{col_w}} {'Old':>10}  {'New':>10}  {'Delta'}")
        print(f"  {'-'*col_w} {'-'*10}  {'-'*10}  {'-'*16}")
        print(f"  {'Bot cost (estimated)':<{col_w}} ${o_bot:>9.4f}  ${n_bot:>9.4f}  {_delta_str(o_bot, n_bot, higher_is_better=False)}")
        print(f"  {'Judge cost':<{col_w}} ${o_judge:>9.4f}  ${n_judge:>9.4f}  {_delta_str(o_judge, n_judge, higher_is_better=False)}")
        print(f"  {'TOTAL':<{col_w}} ${o_total:>9.4f}  ${n_total:>9.4f}  {_delta_str(o_total, n_total, higher_is_better=False)}")
        print(f"  Note: Bot costs estimated (4 chars ≈ 1 token). Judge costs exact.")

    print(f"\n{'=' * W}")


# ─── Markdown report ──────────────────────────────────────────────────────────


def save_markdown_comparison(old: Dict, new: Dict, old_path: str, new_path: str) -> Path:
    sydney_tz = ZoneInfo("Australia/Sydney")
    now_sydney = datetime.now(sydney_tz)
    filename = f"comparison_report_{now_sydney.strftime('%Y%m%d_%H%M%S')}_AEST.md"
    path = ERROR_ANALYSIS_DIR / filename

    old_meta = old["metadata"]
    new_meta = new["metadata"]
    old_st = old["single_turn"]
    new_st = new["single_turn"]
    old_mt = old["multi_turn"]
    new_mt = new["multi_turn"]

    lines: List[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"\n{'#' * level} {text}\n")

    def p(text: str = "") -> None:
        lines.append(text)

    def table_row(*cols: Any) -> str:
        return "| " + " | ".join(str(c) for c in cols) + " |"

    def table_sep(*widths: int) -> str:
        return "| " + " | ".join("-" * w for w in widths) + " |"

    # ── Title ──────────────────────────────────────────────────────────────────
    lines.append("# JO's Bike Shop Chatbot — Baseline Comparison Report")
    p(f"> **Comparison run (Sydney):** {now_sydney.strftime('%A, %d %B %Y %H:%M:%S %Z')}  ")
    p(f"> **Old baseline:** `{old_path}`  ")
    p(f"> **New baseline:** `{new_path}`  ")
    p(f"> **Old chatbot model:** `{old_meta.get('chatbot_model', '?')}`  &nbsp; "
      f"**Old judge:** `{old_meta.get('judge_model', '?')}`  ")
    p(f"> **New chatbot model:** `{new_meta.get('chatbot_model', '?')}`  &nbsp; "
      f"**New judge:** `{new_meta.get('judge_model', '?')}`  ")

    if old_meta.get("judge_model") != new_meta.get("judge_model"):
        p(f"\n> ⚠️ **Warning:** Judge models differ (`{old_meta.get('judge_model')}` vs "
          f"`{new_meta.get('judge_model')}`). Score changes may reflect judge calibration "
          f"differences rather than chatbot quality changes. Keep the judge model fixed "
          f"when comparing chatbot models.")

    # ── Executive Summary ──────────────────────────────────────────────────────
    h(2, "Executive Summary")

    p("### Single-Turn Performance\n")
    p("| Metric | Old | New | Delta |")
    p(table_sep(34, 12, 12, 20))
    st_metrics = [
        ("Mode Detection Accuracy",  "mode_detection_accuracy",  True),
        ("Avg Answer Relevancy",     "avg_answer_relevancy",     True),
        ("Avg Task Completion",      "avg_task_completion",      True),
        ("Avg Conversation Quality", "avg_conversation_quality", True),
    ]
    for label, key, hib in st_metrics:
        o = old_st.get(key)
        n = new_st.get(key)
        old_fmt = _pct(o) if key == "mode_detection_accuracy" else _score_badge(o)
        new_fmt = _pct(n) if key == "mode_detection_accuracy" else _score_badge(n)
        p(table_row(label, old_fmt, new_fmt, _delta_badge(o, n, hib)))

    o_d = old_st.get("overall_success_distribution", {})
    n_d = new_st.get("overall_success_distribution", {})
    p(table_row(
        "Pass / Partial / Fail",
        f"{o_d.get('2',0)} / {o_d.get('1',0)} / {o_d.get('0',0)}",
        f"{n_d.get('2',0)} / {n_d.get('1',0)} / {n_d.get('0',0)}",
        "",
    ))

    p("\n### Multi-Turn Performance\n")
    p("| Metric | Old | New | Delta |")
    p(table_sep(34, 12, 12, 20))
    mt_metrics = [
        ("Avg Conversation Completeness", "avg_conversation_completeness", True),
        ("Avg Knowledge Retention",       "avg_knowledge_retention",       True),
        ("Avg Role Adherence",            "avg_role_adherence",            True),
    ]
    for label, key, hib in mt_metrics:
        o = old_mt.get(key)
        n = new_mt.get(key)
        p(table_row(label, _score_badge(o), _score_badge(n), _delta_badge(o, n, hib)))

    o_d = old_mt.get("overall_success_distribution", {})
    n_d = new_mt.get("overall_success_distribution", {})
    p(table_row(
        "Pass / Partial / Fail",
        f"{o_d.get('2',0)} / {o_d.get('1',0)} / {o_d.get('0',0)}",
        f"{n_d.get('2',0)} / {n_d.get('1',0)} / {n_d.get('0',0)}",
        "",
    ))

    # ── Top failure mode changes ───────────────────────────────────────────────
    total_old = old_st["total_scenarios"] + old_mt["total_scenarios"]
    total_new = new_st["total_scenarios"] + new_mt["total_scenarios"]
    old_fm_combined = {
        k: old_st["failure_mode_frequencies"].get(k, 0) + old_mt["failure_mode_frequencies"].get(k, 0)
        for k in FAILURE_MODES
    }
    new_fm_combined = {
        k: new_st["failure_mode_frequencies"].get(k, 0) + new_mt["failure_mode_frequencies"].get(k, 0)
        for k in FAILURE_MODES
    }
    fm_rows = sorted(
        [(k, old_fm_combined[k], new_fm_combined[k]) for k in FAILURE_MODES],
        key=lambda x: abs(x[2] - x[1]),
        reverse=True,
    )
    changed_fm = [(k, o, n) for k, o, n in fm_rows if o != n]

    p("\n### Failure Mode Changes\n")
    if changed_fm:
        p("| Failure Mode | Old | New | Change |")
        p(table_sep(42, 7, 7, 16))
        for k, o, n in changed_fm[:8]:
            change = n - o
            better = change < 0
            sign = "+" if change > 0 else ""
            icon = "🟢" if better else "🔴"
            p(table_row(k, f"{o}/{total_old}", f"{n}/{total_new}",
                        f"{icon} {sign}{change} ({'better' if better else 'worse'})"))
    else:
        p("_No change in failure mode frequencies._")

    # ── Cost comparison ────────────────────────────────────────────────────────
    old_cost = old_meta.get("cost")
    new_cost = new_meta.get("cost")
    if old_cost and new_cost:
        h(2, "Cost Comparison")
        p("| Component | Old | New | Delta |")
        p(table_sep(30, 14, 14, 20))
        cost_rows: List[Tuple[str, float, float, bool]] = [
            ("Bot cost (estimated)", old_cost["bot"]["cost_usd_estimated"], new_cost["bot"]["cost_usd_estimated"], False),
            ("Judge cost (total)",   old_cost["judge"]["total_judge_cost_usd"], new_cost["judge"]["total_judge_cost_usd"], False),
            ("**Total cost**",       old_cost["total_cost_usd"], new_cost["total_cost_usd"], False),
        ]
        for label, o, n, hib in cost_rows:
            p(table_row(label, f"${o:.4f}", f"${n:.4f}", _delta_badge(o, n, hib)))
        p(f"\n> _Bot token counts are estimated (len/4). "
          f"Judge costs are exact from DeepEval and OpenAI usage responses._\n")

    # ── Detailed metric comparison ─────────────────────────────────────────────
    h(2, "Detailed Single-Turn Comparison")
    p("Per-trace comparison using the scenario tuples as identifiers. "
      "Old and new runs may have different per-trace scores if the simulated "
      "conversations differed, but the scenario tuples are fixed.\n")

    old_st_traces = {t["test_id"]: t for t in old_st.get("per_trace", [])}
    new_st_traces = {t["test_id"]: t for t in new_st.get("per_trace", [])}
    all_st_ids = sorted(set(old_st_traces) | set(new_st_traces))

    if all_st_ids:
        p("| ID | Scenario | AR old→new | TC old→new | CQ old→new | Mode ✓ |")
        p(table_sep(6, 42, 16, 16, 16, 8))
        for tid in all_st_ids:
            old_t = old_st_traces.get(tid, {})
            new_t = new_st_traces.get(tid, {})
            sc = (old_t.get("scenario") or new_t.get("scenario") or "?")[:42]

            def metric_delta(key: str) -> str:
                o = (old_t.get("metrics") or {}).get(key, {}).get("score")
                n = (new_t.get("metrics") or {}).get(key, {}).get("score")
                if o is None and n is None:
                    return "n/a"
                o_s = _fmt(o) if o is not None else "n/a"
                n_s = _fmt(n) if n is not None else "n/a"
                delta = _delta_badge(o, n) if o is not None and n is not None else ""
                return f"{o_s}→{n_s} {delta}"

            mode_old = "✓" if old_t.get("mode_correct") else ("✗" if "mode_correct" in old_t else "?")
            mode_new = "✓" if new_t.get("mode_correct") else ("✗" if "mode_correct" in new_t else "?")
            p(table_row(tid, sc,
                        metric_delta("Answer Relevancy"),
                        metric_delta("Task Completion"),
                        metric_delta("Conversation Quality"),
                        f"{mode_old}→{mode_new}"))
        p("\n_AR = Answer Relevancy · TC = Task Completion · CQ = Conversation Quality_\n")

    h(2, "Detailed Multi-Turn Comparison")
    p("Note: multi-turn scores have natural run-to-run variance because the "
      "ConversationSimulator generates different user utterances each run. "
      "Treat deltas of < 0.05 as noise.\n")

    old_mt_traces = {t["test_id"]: t for t in old_mt.get("per_trace", [])}
    new_mt_traces = {t["test_id"]: t for t in new_mt.get("per_trace", [])}
    all_mt_ids = sorted(set(old_mt_traces) | set(new_mt_traces))

    if all_mt_ids:
        p("| ID | Scenario | CC old→new | KR old→new | RA old→new |")
        p(table_sep(6, 42, 18, 18, 18))
        for tid in all_mt_ids:
            old_t = old_mt_traces.get(tid, {})
            new_t = new_mt_traces.get(tid, {})
            sc = (old_t.get("scenario") or new_t.get("scenario") or "?")[:42]

            def mt_delta(key: str) -> str:
                o = (old_t.get("metrics") or {}).get(key, {}).get("score")
                n = (new_t.get("metrics") or {}).get(key, {}).get("score")
                if o is None and n is None:
                    return "n/a"
                o_s = _fmt(o) if o is not None else "n/a"
                n_s = _fmt(n) if n is not None else "n/a"
                delta = _delta_badge(o, n) if o is not None and n is not None else ""
                return f"{o_s}→{n_s} {delta}"

            p(table_row(tid, sc,
                        mt_delta("Conversation Completeness"),
                        mt_delta("Knowledge Retention"),
                        mt_delta("Role Adherence")))
        p("\n_CC = Conversation Completeness · KR = Knowledge Retention · RA = Role Adherence_\n")

    # ── Full failure mode table ────────────────────────────────────────────────
    h(2, "Failure Mode Analysis")
    p("Full breakdown across all 12 failure modes. Fewer occurrences = better.\n")

    p("| Failure Mode | Old (ST+MT) | New (ST+MT) | Change |")
    p(table_sep(42, 14, 14, 20))
    for k, o, n in fm_rows:
        change = n - o
        better = change < 0
        sign = "+" if change > 0 else ""
        icon = ("🟢" if better else "🔴") if change != 0 else "➡"
        change_str = f"{icon} {sign}{change}" if change != 0 else "➡ 0"
        p(table_row(k, f"{o}/{total_old}", f"{n}/{total_new}", change_str))

    p()
    h(3, "Failure Mode Definitions")
    for k, desc in FAILURE_MODES.items():
        p(f"**{k}:** {desc}  ")

    # ── Footer ─────────────────────────────────────────────────────────────────
    h(2, "How to Interpret This Report")
    p("- **🟢 green delta** — the new model improved on this metric")
    p("- **🔴 red delta** — the new model regressed on this metric")
    p("- **➡ 0** — no change")
    p("- Multi-turn metric deltas < 0.05 are within expected run-to-run noise")
    p("- If judge models differ, score changes are not purely attributable to the chatbot model swap")
    p(f"\n---\n_Generated by `compare_baselines.py` · {now_sydney.strftime('%d %b %Y %H:%M %Z')}_")

    path.write_text("\n".join(lines))
    return path


# ─── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two baseline_scores.json files and report metric deltas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python compare_baselines.py old.json new.json\n"
            "  python compare_baselines.py --old baseline_gpt4o.json --new baseline_gemma4.json"
        ),
    )
    parser.add_argument("positional", nargs="*", help="Old and new baseline files (positional shorthand)")
    parser.add_argument("--old", help="Path to the old (reference) baseline_scores.json")
    parser.add_argument("--new", help="Path to the new (comparison) baseline_scores.json")
    args = parser.parse_args()

    # Accept both positional and named args
    if args.positional and len(args.positional) == 2:
        old_path, new_path = args.positional
    elif args.old and args.new:
        old_path, new_path = args.old, args.new
    else:
        parser.print_help()
        sys.exit(1)

    old = _load(old_path)
    new = _load(new_path)

    print_comparison(old, new)

    ERROR_ANALYSIS_DIR.mkdir(exist_ok=True)
    report_path = save_markdown_comparison(old, new, old_path, new_path)
    print(f"  Markdown comparison report saved to: {report_path}")


if __name__ == "__main__":
    main()
