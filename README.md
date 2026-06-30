# Spacecraft Telemetry Health Assessment

Ground Segment / Operations — Internship Technical Assessment

## Quick Start

```bash
pip install -r requirements.txt

# Generate synthetic telemetry data
python data/synthetic_generator.py

# Inject anomalies into the dataset
python data/anomaly_injection.py

# Run the operator UI
streamlit run ui/app.py
```

## Architecture

telemetry + operator note → fused decision support → human operator

The system ingests per-pass telemetry from a 3U–6U CubeSat-class LEO satellite
together with the operator's free-text note, and produces a severity-graded
health assessment with reasoning an operator can act on.

**This is decision support, not autonomous flight software.**
A human operator confirms every action.

## Project Structure

```
telemetry-health/
├── data/                  # Synthetic data generation
│   ├── synthetic_generator.py
│   ├── anomaly_injection.py
│   └── generated/         # Output data files
├── pipeline/              # Core assessment pipeline
│   ├── hard_limits.py     # Independent rule engine (always wins)
│   ├── numeric_scoring.py # Isolation Forest / One-Class SVM scoring
│   ├── text_scoring.py    # Operator note scoring
│   ├── fusion.py          # Late fusion of numeric + text
│   └── severity.py        # Severity band definitions
├── experiments/           # Architecture-level comparisons
│   ├── fusion_topology_compare.py
│   └── granularity_compare.py
├── ui/
│   └── app.py             # Streamlit operator interface
├── tests/                 # Unit tests
├── report/
│   └── report.md
├── requirements.txt
└── README.md
```
