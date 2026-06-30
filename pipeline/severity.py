"""
Severity band definitions for the spacecraft telemetry health assessment.

Maps to PDD Section 4 — four-level severity scheme:
  NOMINAL → WATCH → CAUTION → CRITICAL

These thresholds are applied to the normalised anomaly scores (0.0 to 1.0)
produced by the numeric and text scoring layers.
"""

# --- Severity levels (PDD Section 4) ---
NOMINAL = "NOMINAL"
WATCH = "WATCH"
CAUTION = "CAUTION"
CRITICAL = "CRITICAL"

# Ordered from lowest to highest severity
SEVERITY_ORDER = [NOMINAL, WATCH, CAUTION, CRITICAL]

# --- Score thresholds for mapping continuous anomaly scores to severity ---
# These define ranges over the normalised 0-1 anomaly score scale.
# Hard-limit breaches bypass this entirely and go straight to CRITICAL.
SCORE_THRESHOLDS = {
    NOMINAL: (0.0, 0.3),    # Score below 0.3 → all within expected bands
    WATCH:   (0.3, 0.6),    # Score 0.3-0.6 → trending toward a limit
    CAUTION: (0.6, 0.85),   # Score 0.6-0.85 → soft limit breached
    CRITICAL: (0.85, 1.0),  # Score above 0.85 → high-confidence active fault
}

# Colours for the UI severity badges (PDD Section 7.3)
SEVERITY_COLOURS = {
    NOMINAL: "#2ecc71",   # green
    WATCH:   "#f39c12",   # amber
    CAUTION: "#e67e22",   # orange
    CRITICAL: "#e74c3c",  # red
}


def score_to_severity(score):
    """Convert a normalised anomaly score (0.0–1.0) to a severity level.

    Args:
        score: float in [0.0, 1.0]

    Returns:
        One of NOMINAL, WATCH, CAUTION, CRITICAL
    """
    if score >= SCORE_THRESHOLDS[CRITICAL][0]:
        return CRITICAL
    elif score >= SCORE_THRESHOLDS[CAUTION][0]:
        return CAUTION
    elif score >= SCORE_THRESHOLDS[WATCH][0]:
        return WATCH
    else:
        return NOMINAL


def higher_severity(a, b):
    """Return whichever severity level is more severe."""
    return a if SEVERITY_ORDER.index(a) >= SEVERITY_ORDER.index(b) else b
