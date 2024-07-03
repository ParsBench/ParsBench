from parsbench.scores.base import wrap_scorer


@wrap_scorer
def exact_match(completion: str, target: str) -> int:
    return int(completion == target)
