from pathlib import Path

try:
    import math_equivalence as math_metric
except ImportError:
    raise Exception(
        "The math_equivalence package is not installed."
        "You should install it manually by `pip install git+https://github.com/hendrycks/math.git`"
    )


from parsbench.scores.base import Scorer, wrap_scorer
from parsbench.tasks.base import (
    JSONLineDataLoader,
    LazyLoadTemplates,
    PromptTemplate,
    Task,
    TaskCategory,
    TaskMatchGroup,
)

PROMPTS_PATH = Path(__file__).parent / "prompts"
DATA_URL = "https://raw.githubusercontent.com/Ipouyall/Benchmarking_ChatGPT_for_Persian/main/Benchmark/Math/math.jsonl"


@wrap_scorer
def math_equivalence(completion: str, target: str):
    return math_metric.is_equiv(completion, target)


def _preserve_digit(input_string):
    end_idx = len(input_string) - 1
    while end_idx >= 0 and not input_string[end_idx].isdigit():
        end_idx -= 1
    start_idx = end_idx

    while (
        input_string[start_idx].isdigit()
        or input_string[start_idx] in [".", ",", "/", "\\", "{", "}"]
    ) and start_idx > 0:
        start_idx -= 1

    return input_string[start_idx + 1 : end_idx + 1]


class PersianMath(Task):
    task_name: str = "Persian Math"
    task_category: TaskCategory = TaskCategory.MATH

    data_loader: JSONLineDataLoader = JSONLineDataLoader(data_path=DATA_URL)
    data_target_key: str = "target"

    prompt_template: PromptTemplate = PromptTemplate(
        language_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_math.txt",
            fa=PROMPTS_PATH / "fa_math.txt",
        ),
        prompt_shot_examples={
            "en": LazyLoadTemplates(
                {
                    1: PROMPTS_PATH / "en_math_1_shot.txt",
                    3: PROMPTS_PATH / "en_math_3_shot.txt",
                }
            ),
            "fa": LazyLoadTemplates(
                {
                    1: PROMPTS_PATH / "fa_math_1_shot.txt",
                    3: PROMPTS_PATH / "fa_math_3_shot.txt",
                }
            ),
        },
        prompt_variables_mapping={
            "problem": "problem",
        },
        target_variables_mapping={"target": "target"},
    )

    scorer: Scorer = math_equivalence

    def score_matches(self, matches: TaskMatchGroup) -> TaskMatchGroup:
        def _format_completion(completion):
            idx = max(completion.rfind("[پاسخ]"), completion.rfind("[answer]"))
            completion = completion[idx:]
            return _preserve_digit(completion)

        matches.format_completions(_format_completion)

        return super().score_matches(matches)
