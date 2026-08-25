# Scores

Class representing a Scorer object.

Attributes:

| Name   | Type                          | Description                         |
| ------ | ----------------------------- | ----------------------------------- |
| `func` | `Callable[[str, str], float]` | The scoring function to be wrapped. |

Methods:

| Name      | Description                                                                                                                    |
| --------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `measure` | str, target: str) -> float: Calculates the score between the completion and target strings using the wrapped scoring function. |
| `name`    | Returns the name of the wrapped scoring function with underscores replaced by spaces and title-cased.                          |

Source code in `parsbench/scores/base.py`

```
class Scorer:
    """
    Class representing a Scorer object.

    Attributes:
        func (Callable[[str, str], float]): The scoring function to be wrapped.

    Methods:
        measure(completion: str, target: str) -> float:
            Calculates the score between the completion and target strings using the wrapped scoring function.

        name() -> str:
            Returns the name of the wrapped scoring function with underscores replaced by spaces and title-cased.
    """

    def __init__(self, func: Callable[[str, str], float]):
        self.func = func

    def measure(self, completion: str, target: str) -> float:
        return self.func(completion, target)

    @property
    def name(self) -> str:
        return self.func.__name__.replace("_", " ").title()
```

Wraps a scorer function inside the Scorer class.

Source code in `parsbench/scores/base.py`

```
def wrap_scorer(func):
    """Wraps a scorer function inside the Scorer class."""
    return Scorer(func)
```

Source code in `parsbench/scores/common.py`

```
@wrap_scorer
def exact_match(completion: str, target: str) -> int:
    return int(completion == target)
```

Source code in `parsbench/scores/machine_translation.py`

```
@wrap_scorer
def english_sentence_bleu(completion: str, target: str) -> float:
    _ensure_punkt()

    reference_translation = [nltk.word_tokenize(target)]
    model_translation = nltk.word_tokenize(completion)
    bleu_score = nltk.translate.bleu(
        reference_translation, model_translation, weights=(1,)
    )
    return bleu_score
```

Source code in `parsbench/scores/machine_translation.py`

```
@wrap_scorer
def persian_sentence_bleu(completion: str, target: str) -> float:
    reference_translation = [hazm.word_tokenize(target)]
    model_translation = hazm.word_tokenize(completion)
    bleu_score = nltk.translate.bleu(
        reference_translation, model_translation, weights=(1,)
    )
    return bleu_score
```

Source code in `parsbench/scores/summarization.py`

```
@wrap_scorer
def english_rouge(completion: str, target: str) -> float:
    scores = _english_scorer().score(target, completion)
    return scores["rouge1"].fmeasure
```

Source code in `parsbench/scores/summarization.py`

```
@wrap_scorer
def persian_rouge(completion: str, target: str) -> float:
    scores = _persian_scorer().score(target, completion)
    return scores["rouge1"].fmeasure
```
