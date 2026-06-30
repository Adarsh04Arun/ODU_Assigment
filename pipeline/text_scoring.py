"""
Text scoring layer — lightweight classification of operator notes.

Implements PDD Section 6.3 (using the operator's text note).

Design decision: keyword/phrase-pattern classifier, not TF-IDF + logistic
regression. Rationale (per Implementation Plan Step 6.2): with a hand-built
synthetic note corpus, data volume is too small to honestly validate a learned
model. The keyword approach is the easier of the two to explain line-by-line
in a live demo, and "I chose the simpler option because I could fully defend
it" is a stronger answer than a marginally fancier model you'd have to hedge on.

The operator note vocabulary is narrow and repetitive (PDD Section 6.3):
operators use a small set of concern-indicating and reassurance-indicating
phrases. A hand-built keyword classifier captures this well.
"""

# ===========================================================================
# 6.1 — Operator note vocabulary (concern and reassurance phrases)
# ===========================================================================

# Concern-indicating phrases — presence pushes severity up
CONCERN_PHRASES = {
    # High concern (maps toward CAUTION)
    "critical": 0.9,
    "emergency": 0.9,
    "failure": 0.85,
    "failed": 0.85,
    "loss of": 0.8,
    "unresponsive": 0.8,
    "unexpected": 0.7,
    "anomaly": 0.7,
    "anomalous": 0.7,
    "out of range": 0.7,
    "breach": 0.7,
    "exceeded": 0.65,
    "spike": 0.65,
    "degraded": 0.6,
    "degrading": 0.6,
    # Medium concern (maps toward WATCH)
    "dipping": 0.5,
    "dropping": 0.5,
    "rising": 0.45,
    "climbing": 0.45,
    "trending": 0.4,
    "drift": 0.4,
    "drifting": 0.4,
    "fluctuating": 0.4,
    "watching it": 0.4,
    "keep an eye": 0.4,
    "monitor": 0.35,
    "monitoring": 0.35,
    "slightly off": 0.35,
    "below expected": 0.35,
    "above expected": 0.35,
    "weaker than usual": 0.35,
    "higher than usual": 0.35,
    "lower than usual": 0.35,
    # Low concern
    "within tolerance": 0.2,
    "borderline": 0.3,
    "marginal": 0.3,
}

# Reassurance-indicating phrases — presence pushes severity down
REASSURANCE_PHRASES = {
    "nominal": -0.4,
    "all nominal": -0.5,
    "looks good": -0.4,
    "looks fine": -0.4,
    "all good": -0.4,
    "stable": -0.3,
    "normal": -0.3,
    "as expected": -0.3,
    "within limits": -0.3,
    "no issues": -0.4,
    "no concerns": -0.4,
    "healthy": -0.3,
    "clean pass": -0.5,
    "routine": -0.3,
}


def score_note(note_text):
    """Score an operator note for severity contribution.

    Args:
        note_text: string, the operator's free-text note for this pass.

    Returns:
        dict with keys:
            "score":      float in [0, 1] (0 = reassuring, 1 = concerning)
            "confidence": float in [0, 1] (based on how many phrases matched)
            "matched_phrases": list of (phrase, weight) tuples that fired
            "reasoning":  human-readable string explaining the score
    """
    text = note_text.lower().strip()
    matched = []
    raw_score = 0.0

    # Check concern phrases
    for phrase, weight in CONCERN_PHRASES.items():
        if phrase in text:
            matched.append((phrase, weight))
            raw_score += weight

    # Check reassurance phrases
    for phrase, weight in REASSURANCE_PHRASES.items():
        if phrase in text:
            matched.append((phrase, weight))
            raw_score += weight  # weight is negative for reassurance

    # Normalise to [0, 1]
    # With multiple matches, raw_score can exceed 1.0 or go below 0.0
    anomaly_score = max(0.0, min(1.0, raw_score))

    # Confidence: based on number of matched phrases
    # 0 matches → very low confidence (no signal from text)
    # 1 match → medium confidence
    # 2+ matches → higher confidence
    if len(matched) == 0:
        confidence = 0.1  # almost no confidence — text gave us nothing
    elif len(matched) == 1:
        confidence = 0.5
    else:
        confidence = min(1.0, 0.5 + len(matched) * 0.15)

    # Build reasoning string
    if not matched:
        reasoning = "No recognised concern or reassurance phrases found in note."
    else:
        parts = []
        for phrase, weight in sorted(matched, key=lambda x: abs(x[1]), reverse=True):
            direction = "concern" if weight > 0 else "reassurance"
            parts.append(f"'{phrase}' ({direction}, weight={weight:+.2f})")
        reasoning = "Matched phrases: " + "; ".join(parts)

    return {
        "score": round(anomaly_score, 4),
        "confidence": round(confidence, 4),
        "matched_phrases": matched,
        "reasoning": reasoning,
    }


# ===========================================================================
# 6.1 — Synthetic operator note corpus
# Notes paired with expected severity for validation.
# ===========================================================================

SAMPLE_NOTES = [
    # NOMINAL notes
    ("All systems nominal. Clean pass, no issues observed.", "NOMINAL"),
    ("Routine downlink pass. All parameters within limits. Looks good.", "NOMINAL"),
    ("Nominal housekeeping. Battery stable, temps normal.", "NOMINAL"),
    ("Clean pass. All subsystems healthy, as expected.", "NOMINAL"),
    ("Standard contact. No concerns. All nominal.", "NOMINAL"),

    # WATCH notes
    ("Battery voltage dipping slightly during eclipse. Watching it.", "WATCH"),
    ("RSSI weaker than usual but within tolerance. Will monitor next pass.", "WATCH"),
    ("Temperature trending upward on panel B. Still in range, monitoring.", "WATCH"),
    ("Slight drift in reaction wheel speed. Keep an eye on it.", "WATCH"),
    ("CPU load higher than usual. Within limits but watching it.", "WATCH"),
    ("Battery dipping a bit during eclipse, slightly off from last pass.", "WATCH"),
    ("Memory occupancy climbing slowly. Monitoring trend.", "WATCH"),

    # CAUTION notes
    ("Unexpected temperature spike on panel B. Investigating.", "CAUTION"),
    ("Battery voltage dropping faster than eclipse should explain. Anomalous.", "CAUTION"),
    ("RSSI degraded significantly. Signal weaker than expected for this geometry.", "CAUTION"),
    ("Anomaly in OBC telemetry. CPU load spike with unexpected memory jump.", "CAUTION"),
    ("Reaction wheel speed exceeded normal range briefly. Checking attitude.", "CAUTION"),

    # CRITICAL notes
    ("Battery failure. Voltage dropping to critical levels. Emergency.", "CRITICAL"),
    ("Loss of attitude control. Tumble detected. Critical anomaly.", "CRITICAL"),
    ("Complete loss of signal. Satellite unresponsive.", "CRITICAL"),
]


if __name__ == "__main__":
    # Deliverable check: can we explain every score?
    print("=== Text Scoring Layer — Deliverable Check ===\n")
    for note_text, expected_sev in SAMPLE_NOTES:
        result = score_note(note_text)
        print(f"Note: \"{note_text[:60]}...\"")
        print(f"  Score: {result['score']:.2f}  Confidence: {result['confidence']:.2f}")
        print(f"  {result['reasoning']}")
        print(f"  Expected: {expected_sev}")
        print()
