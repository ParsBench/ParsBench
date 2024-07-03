import hazm
import nltk

from .base import wrap_scorer


@wrap_scorer
def english_sentence_bleu(completion: str, target: str) -> float:
    reference_translation = [nltk.word_tokenize(target)]
    model_translation = nltk.word_tokenize(completion)
    bleu_score = nltk.translate.bleu(
        reference_translation, model_translation, weights=(1,)
    )
    return bleu_score


@wrap_scorer
def persian_sentence_bleu(completion: str, target: str) -> float:
    nltk.download("punkt", quiet=True)

    reference_translation = [hazm.word_tokenize(target)]
    model_translation = hazm.word_tokenize(completion)
    bleu_score = nltk.translate.bleu(
        reference_translation, model_translation, weights=(1,)
    )
    return bleu_score
