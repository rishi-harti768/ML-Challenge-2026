# Business Entity Resolution (ML Challenge 2026)

Matches noisy business records describing the same real-world business across three independent, differently-formatted data sources, scored by how precisely and completely those matches are found.

## Language

**Source 1 / Source 2 / Source 3**:
The three independent record sets, distinguished by their `entity_id` prefix (`S1-`/`S2-`/`S3-`) and by which file they appear in. Source 1 is the deduplicated reference source: every Source 1 entity gets exactly one output row, listing which Source 2 and/or Source 3 records refer to the same business.
_Avoid_: source, dataset, table (when a specific one of the three is meant, name it).

**Reference entity**:
A Source 1 record — the fixed point each prediction is anchored to. "Match" always means "a Source 2/3 record matched to a reference entity," never the reverse.
_Avoid_: anchor, seed record.

**Singleton**:
A reference entity with no true matches in Source 2 or Source 3. Predicting an empty match list for one scores full credit (1.0); predicting any match for one scores 0.0. Not a data-quality problem — a valid, expected outcome the pipeline must recognize as an outcome, not an error.
_Avoid_: unmatched entity, orphan.

**Blocking / candidate generation**:
The stage that narrows the full Source 2 × Source 3 space down to a plausible candidate set per reference entity, before any matching model runs. Sets the recall ceiling for everything downstream — a true match blocking never proposes can never be recovered later.
_Avoid_: pre-filtering, indexing (unless referring to a specific technique like an ANN index).

**Candidate pair**:
One (reference entity, Source 2/3 record) pair that survived blocking and was handed to the matching model for a decision. The full set of these, per reference entity, is `candidate_pairs.tsv` — the exact input the matching model scored, not raw blocking output that was later filtered further.
_Avoid_: candidate match, blocking result.

**Matching model**:
The stage that decides, per candidate pair, whether the two records refer to the same business. Its output — the surviving pairs — is `matching_results.tsv`, the only file scored on the leaderboard.
_Avoid_: classifier (only use when the matching model literally is one), scorer.

**Macro-average F_0.5**:
The competition metric: F_0.5 (precision weighted 2x over recall) computed per reference entity, then averaged across all reference entities, singletons included. Distinguishes this from a pooled/micro F_0.5 over all pairs, which the leaderboard does not use.
_Avoid_: F-score, F1 (this competition never uses F1).

**Random holdout**:
A validation split built by sampling reference entities uniformly at random from the training set, regardless of country. Measures matching quality under the same country mix (US/India) the model trained on — it does not test generalization to an unseen country.
_Avoid_: held-out validation split (ambiguous between this and country-held-out), dev set.

**Country-held-out validation**:
A validation split built by training with one training country entirely excluded (e.g. India), then measuring matching quality only on that excluded country's reference entities. Simulates the test set's France shift, since France is likewise absent from training. A large gap between this score and the random holdout score signals the model or blocking is leaning on country-specific patterns rather than generic string similarity.
_Avoid_: unseen-country split, generalization split.

**Domain shift / unseen country**:
A country label present in the test set but absent from training (currently: `France`). The pipeline must handle it without country-specific hardcoding, since `country` is an open-set label, not a fixed enum.
_Avoid_: out-of-distribution (too general — here it specifically means an unseen `country` value).
