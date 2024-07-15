import hazm
import nltk
from rouge_score import rouge_scorer

from .base import wrap_scorer


@wrap_scorer
def english_rouge(completion: str, target: str) -> float:
    nltk.download("punkt", quiet=True)

    tokenizer = nltk.tokenize.NLTKWordTokenizer()

    scorer = rouge_scorer.RougeScorer(["rouge1"], tokenizer=tokenizer)
    scores = scorer.score(target, completion)
    return scores["rouge1"].fmeasure


@wrap_scorer
def persian_rouge(completion: str, target: str) -> float:
    tokenizer = hazm.WordTokenizer()

    scorer = rouge_scorer.RougeScorer(["rouge1"], tokenizer=tokenizer)
    scores = scorer.score(target, completion)
    return scores["rouge1"].fmeasure
