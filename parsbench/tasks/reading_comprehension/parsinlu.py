from pathlib import Path

from hazm import Normalizer

from parsbench.scores.base import Scorer, wrap_scorer
from parsbench.tasks.base import (
    JSONLineDataLoader,
    LazyLoadTemplates,
    PromptTemplate,
    Task,
    TaskCategory,
)

PROMPTS_PATH = Path(__file__).parent / "prompts"
DATA_URL = "https://raw.githubusercontent.com/persiannlp/parsinlu/master/data/reading_comprehension/eval.jsonl"


def _preprocess_text(s, normalizer):
    def normalize(text):
        text.replace("پاسخ:", "").replace("جواب:", "").replace("answer:", "")
        return normalizer.normalize(text)

    def remove_punc_stopword(text):
        exclude = ["?", ".", "!", "؟", ":", "،", ")", "(", "..."]
        return "".join(ch for ch in text if ch not in exclude)

    return normalize(remove_punc_stopword(s))


def _compute_f1(prediction, truth):
    pred_tokens = prediction.split()
    truth_tokens = truth.split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


@wrap_scorer
def common_tokens(completion: str, target: list[dict]) -> float:
    normalizer = Normalizer()

    model_ans = _preprocess_text(completion, normalizer)
    correct_ans = [_preprocess_text(t[1], normalizer) for t in target]
    f1 = max(_compute_f1(model_ans, answer) for answer in correct_ans)

    return f1


class _ReadingComprehensionPromptTemplate(PromptTemplate):
    def get_target_variables(self, data: dict) -> dict:
        mapped_data = {}
        for tk, dk in self.target_variables_mapping.items():
            if dk not in data:
                raise ValueError(f"Key {dk} not in data.")
            if dk == "answers":
                # Just one of the desired targets is enough for the example shot.
                mapped_data[tk] = data[dk][0][1]
            else:
                mapped_data[tk] = data[dk]
        return mapped_data


class ParsiNLUReadingComprehension(Task):
    task_name: str = "ParsiNLU Reading Comprehension"
    task_category: TaskCategory = TaskCategory.CLASSIC

    data_loader: JSONLineDataLoader = JSONLineDataLoader(data_path=DATA_URL)
    data_target_key: str = "answers"

    prompt_template: PromptTemplate = _ReadingComprehensionPromptTemplate(
        language_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_parsinlu_rc.txt",
            fa=PROMPTS_PATH / "fa_parsinlu_rc.txt",
        ),
        prompt_shot_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_parsinlu_rc_shot.txt",
            fa=PROMPTS_PATH / "fa_parsinlu_rc_shot.txt",
        ),
        prompt_variables_mapping={
            "context": "passage",
            "question": "question",
        },
        target_variables_mapping={"answer": "answers"},
    )

    scorer: Scorer = common_tokens
