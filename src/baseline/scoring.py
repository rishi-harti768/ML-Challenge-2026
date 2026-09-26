"""Macro-average F_0.5, computed exactly the way the leaderboard does it:
per reference entity, then averaged - singletons included, with the
README's explicit singleton rule (correct empty prediction = 1.0, any
false match on a true singleton = 0.0).
"""


def f_beta_entity(true_ids: set, pred_ids: set, beta: float = 0.5) -> float:
    """F_beta for one reference entity."""
    if not true_ids:
        return 1.0 if not pred_ids else 0.0
    if not pred_ids:
        return 0.0
    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def macro_f_beta(true_map: dict, pred_map: dict, entity_ids, beta: float = 0.5) -> float:
    """Macro-average F_beta over `entity_ids` (every id gets exactly one score)."""
    scores = [
        f_beta_entity(true_map.get(eid, set()), pred_map.get(eid, set()), beta=beta)
        for eid in entity_ids
    ]
    return sum(scores) / len(scores) if scores else 0.0
