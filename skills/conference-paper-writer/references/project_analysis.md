# Project analysis
Inspect: source, notebooks, configs, scripts, datasets (or loaders), logs, result files, checkpoints, plots, docs, prior reports. Skip venv/node_modules/caches/build output. Treat README as a claim to verify.
Establish and record (with file paths and line numbers):
- Problem, importance, scope, assumptions. Gap (only if literature supports it). Objectives.
- Architecture: components, data flow, I/O, algorithms/models/frameworks, decisions that matter scientifically.
- Data: source, size, collection/processing, licence/privacy, splits, leakage risks.
- Method: formulation, preprocessing, training/validation/test, hyperparameters, baselines.
- Results: which experiments were actually executed, where stored, metrics and how computed, reproducible from artifacts?, baselines, ablations, statistics.
- Limitations and threats to validity. Environment (language, libraries+versions, OS, hardware) only if documented or detectable.
Novelty: classify as implementation / application to new domain or data / integration / systems contribution / empirical study / methodological improvement / new algorithm / new finding. State principal and secondary contributions, evidence for each, prior work, and limits of the claim. Weak or unclear contribution: say so and recommend realistic improvements in a separate list; do not add experiments or edit code to inflate novelty.
Conflicting values across files: find the source of truth (timestamps, logs, code); else flag.
