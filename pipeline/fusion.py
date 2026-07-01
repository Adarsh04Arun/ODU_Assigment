"""
Late fusion layer — combines numeric and text severity scores.
Stage 4, the combiner. fuse_scores(breaches, numeric, text, note) implements late fusion:
If any hard-limit breach exists → CRITICAL at confidence 1.0, done (overrides all).
Otherwise take the higher severity of numeric vs text; if both independently agree on the same non-trivial band, escalate one level.
Combined confidence = min of the two ("only as confident as our least confident input").
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.severity import (
    score_to_severity, higher_severity,
    NOMINAL, WATCH, CAUTION, CRITICAL, SEVERITY_ORDER,
)


def fuse_scores(hard_limit_breaches, numeric_result, text_result, operator_note):
    """Combine hard-limit, numeric, and text results into a final assessment.

    This is the core fusion function. The pipeline shape is:
      1. Hard limits checked first — if any breach, result is CRITICAL, done.
      2. Otherwise, numeric and text scores are combined via late fusion.

    Args:
        hard_limit_breaches: list of breach dicts from hard_limits.evaluate_hard_limits()
        numeric_result:      dict from numeric_scoring.aggregate_subsystem_scores()
        text_result:         dict from text_scoring.score_note()
        operator_note:       the raw note string (displayed alongside the trace)

    Returns:
        dict with full assessment including reasoning trace.
    """
    reasoning_trace = []

    #Hard-limit check (always wins, never overridden)
    if hard_limit_breaches:
        for breach in hard_limit_breaches:
            reasoning_trace.append({
                "source": "hard_limit",
                "detail": breach["description"],
                "rule": breach["rule"],
                "subsystem": breach["subsystem"],
                "channel": breach["channel"],
                "threshold": breach["threshold"],
                "actual_value": breach["actual_value"],
            })

        return {
            "severity": CRITICAL,
            "confidence": 1.0,  # hard limits are certain by definition
            "reasoning_trace": reasoning_trace,
            "numeric_score": numeric_result["overall_score"],
            "text_score": text_result["score"],
            "operator_note": operator_note,
            "hard_limit_fired": True,
            "summary": _build_summary(CRITICAL, 1.0, reasoning_trace, True),
        }

    # ------------------------------------------------------------------
    # Stage 2: Late fusion of numeric and text scores
    # ------------------------------------------------------------------

    numeric_score = numeric_result["overall_score"]
    numeric_confidence = numeric_result["overall_confidence"]
    text_score = text_result["score"]
    text_confidence = text_result["confidence"]

    # Add numeric contribution to reasoning trace
    numeric_severity = score_to_severity(numeric_score)
    reasoning_trace.append({
        "source": "numeric",
        "severity": numeric_severity,
        "score": numeric_score,
        "confidence": numeric_confidence,
        "worst_subsystem": numeric_result["worst_subsystem"],
        "detail": (f"Numeric scoring: {numeric_severity} "
                   f"(score={numeric_score:.3f}, confidence={numeric_confidence:.3f}, "
                   f"worst subsystem={numeric_result['worst_subsystem']})"),
    })

    # Add text contribution to reasoning trace
    text_severity = score_to_severity(text_score)
    reasoning_trace.append({
        "source": "text",
        "severity": text_severity,
        "score": text_score,
        "confidence": text_confidence,
        "detail": (f"Note scoring: {text_severity} "
                   f"(score={text_score:.3f}, confidence={text_confidence:.3f})"),
        "matched_phrases": text_result.get("matched_phrases", []),
        "reasoning": text_result.get("reasoning", ""),
    })

    # Late fusion rule: take the higher severity of the two.
    # If both independently agree at the same level, escalate one level.
    fused_severity = higher_severity(numeric_severity, text_severity)

    if (numeric_severity == text_severity
            and numeric_severity != NOMINAL
            and numeric_severity != CRITICAL):
        # Both agree on a non-trivial level → escalate by one
        current_idx = SEVERITY_ORDER.index(fused_severity)
        if current_idx < len(SEVERITY_ORDER) - 1:
            fused_severity = SEVERITY_ORDER[current_idx + 1]
            reasoning_trace.append({
                "source": "fusion_escalation",
                "detail": (f"Both numeric and text independently indicate "
                           f"{numeric_severity} — escalating to {fused_severity}"),
            })

    # Combined confidence: minimum of the two (PDD Section 7.2.2)
    fused_confidence = min(numeric_confidence, text_confidence)

    return {
        "severity": fused_severity,
        "confidence": round(fused_confidence, 4),
        "reasoning_trace": reasoning_trace,
        "numeric_score": numeric_score,
        "text_score": text_score,
        "operator_note": operator_note,
        "hard_limit_fired": False,
        "summary": _build_summary(fused_severity, fused_confidence,
                                  reasoning_trace, False),
    }


def _build_summary(severity, confidence, trace, hard_limit_fired):
    """Build a plain-English summary from the reasoning trace.

    This is what Step 8's UI renders directly, and what the brief's
    "reasoning an operator can act on" requirement is testing.
    """
    if hard_limit_fired:
        rules = [t["rule"] for t in trace if t["source"] == "hard_limit"]
        descriptions = [t["detail"] for t in trace if t["source"] == "hard_limit"]
        return (f"CRITICAL — Hard limit breached: {'; '.join(descriptions)}. "
                f"Rule(s): {', '.join(rules)}. "
                f"This is a physics-based safety limit, not a model judgement.")

    parts = []
    for t in trace:
        if t["source"] == "numeric":
            parts.append(f"Numbers say {t['severity']} "
                         f"(worst: {t['worst_subsystem']}, "
                         f"score={t['score']:.2f})")
        elif t["source"] == "text":
            parts.append(f"Note says {t['severity']} "
                         f"(score={t['score']:.2f})")
        elif t["source"] == "fusion_escalation":
            parts.append(t["detail"])

    conf_str = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
    return (f"{severity} (confidence: {conf_str}, {confidence:.2f}) — "
            + "; ".join(parts))


# ===========================================================================
# CLI: test fusion with demo passes
# ===========================================================================

if __name__ == "__main__":
    import json
    from pipeline.hard_limits import evaluate_hard_limits
    from pipeline.numeric_scoring import (
        score_pass, aggregate_subsystem_scores, load_models
    )
    from pipeline.text_scoring import score_note, SAMPLE_NOTES

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "generated")
    models = load_models(os.path.join(data_dir, "trained_models.pkl"))

    # Load demo passes
    with open(os.path.join(data_dir, "demo_passes.json")) as f:
        demo = json.load(f)

    demo_notes = [
        "Battery failure. Voltage dropping to critical levels. Emergency.",
        "All systems nominal. Clean pass, no issues.",
        "Battery dipping slightly. Watching it. Within tolerance still.",
    ]

    print("=== Fusion Layer - Deliverable Check ===\n")
    for i, (entry, note) in enumerate(zip(demo, demo_notes)):
        p = entry["pass"]
        label = entry["label"]

        # Run the full pipeline
        breaches = evaluate_hard_limits(p)
        sub_scores = score_pass(p, models, "iforest")
        agg = aggregate_subsystem_scores(sub_scores)
        text = score_note(note)

        result = fuse_scores(breaches, agg, text, note)

        print(f"--- Pass {i+1}: {p['pass_id']} (expected: {label['expected_severity']}) ---")
        print(f"  Severity: {result['severity']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Hard limit fired: {result['hard_limit_fired']}")
        print(f"  Summary: {result['summary']}")
        print(f"  Reasoning trace:")
        for t in result["reasoning_trace"]:
            print(f"    [{t['source']}] {t['detail']}")
        print()
