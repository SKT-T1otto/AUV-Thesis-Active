# Map diagnostics

Search ends at discovery; execution thereafter is reported separately.
Scoring, proposing and installed guidance execution are separate measurements.
Exact coupled guidance is a conservative path match; safety-modified paths, holds and unknown provenance are separate.
Execution counts use public physical transitions, including the terminal transition. They do not prove causal action attribution.
Compute attempts include failed attempts. Rescore fallback time is a separate d2_fallback phase.
Trace percentiles cover retained attempts; truncation is explicit. Aggregate counts cover the complete episode.
OLD groups and historical outputs may lack diagnostics; blanks mean unavailable, not zero.
CSV files: participation.csv, compute_attempts.csv and fallback_reasons.csv (when applicable).
