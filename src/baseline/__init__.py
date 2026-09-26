"""Fast classical baseline pipeline for ML Challenge 2026 (issue #9).

Modules:
    normalize   - business-name / address normalization and tokenization
    split       - reusable held-out validation split (issue #2 decision)
    blocking    - multi-key inverted-index candidate generation
    features    - similarity feature engineering on candidate pairs
    matching    - matching model (logistic regression) + threshold tuning
    scoring     - macro-average F_0.5 (per reference entity)
    error_attribution - stage-level error attribution report (issue #7 decision)
    pipeline    - CLI entry points tying the stages together
"""
