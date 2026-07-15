from functools import cache

import hazm
import nltk
from rouge_score import rouge_scorer

from .base import wrap_scorer


@cache
def _english_scorer() -> rouge_scorer.RougeScorer:
    nltk.download("punkt", quiet=True)
    tokenizer = nltk.tokenize.NLTKWordTokenizer()
    return rouge_scorer.RougeScorer(["rouge1"], tokenizer=tokenizer)


@cache
def _persian_scorer() -> rouge_scorer.RougeScorer:
    tokenizer = hazm.WordTokenizer()
    return rouge_scorer.RougeScorer(["rouge1"], tokenizer=tokenizer)


@wrap_scorer
def english_rouge(completion: str, target: str) -> float:
    scores = _english_scorer().score(target, completion)
    return scores["rouge1"].fmeasure


@wrap_scorer
def persian_rouge(completion: str, target: str) -> float:
    scores = _persian_scorer().score(target, completion)
    return scores["rouge1"].fmeasure
