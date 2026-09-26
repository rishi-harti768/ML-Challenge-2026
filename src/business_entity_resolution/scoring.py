def entity_f_beta(pred: set[str], true: set[str], beta: float = 0.5) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(true)
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def macro_f_beta(
    predictions: dict[str, set[str]], truth: dict[str, set[str]], beta: float = 0.5
) -> float:
    if not truth:
        return 0.0
    scores = [entity_f_beta(predictions.get(sid, set()), true_set, beta) for sid, true_set in truth.items()]
    return sum(scores) / len(scores)


def parse_matched_ids(raw: str) -> set[str]:
    if not raw:
        return set()
    return {x for x in raw.split(",") if x}
