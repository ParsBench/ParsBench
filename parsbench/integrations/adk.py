"""Google ADK helpers.

ADK's default `response_match_score` is ROUGE-1 with an ASCII-oriented
tokenizer — it collapses on Perso-Arabic script. `persian_response_match`
is the drop-in replacement: token-level F1 over Persian-normalized tokens.
"""

from parsbench.appeval.normalize import normalize


def persian_response_match(reference: str, response: str) -> float:
    ref = normalize(reference).split()
    got = normalize(response).split()
    if not ref or not got:
        return float(ref == got)
    common: dict[str, int] = {}
    for token in ref:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in got:
        if common.get(token, 0) > 0:
            common[token] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(got)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)
