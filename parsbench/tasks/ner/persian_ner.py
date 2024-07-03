from pathlib import Path

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
DATA_URL = "https://raw.githubusercontent.com/Ipouyall/Benchmarking_ChatGPT_for_Persian/main/Benchmark/NER/ner.jsonl"


@wrap_scorer
def ner_exact_match(completion: str, target: str) -> float:
    tags = [
        "o",
        "per",
        "loc",
        "org",
        "product",
        "event",
        "facility",
    ]
    tag2index = {t: i for i, t in enumerate(tags)}

    try:
        completion_ner = eval(completion)
        target_ner = eval(target)
    except:
        return 0

    mapper = lambda t: (t[0], tag2index.get(t[1], -2))
    completion_ner = dict(map(mapper, completion_ner))
    target_ner = dict(map(mapper, target_ner))

    score = sum(
        [1 if target_ner.get(t, -1) == idx else 0 for t, idx in completion_ner.items()]
    ) / len(completion_ner)

    return score


class PersianNER(Task):
    task_name: str = "Persian NER"
    task_category: TaskCategory = TaskCategory.CLASSIC

    data_loader: JSONLineDataLoader = JSONLineDataLoader(data_path=DATA_URL)
    data_target_key: str = "output"

    prompt_template: PromptTemplate = PromptTemplate(
        language_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_persian_ner.txt",
            fa=PROMPTS_PATH / "fa_persian_ner.txt",
        ),
        prompt_shot_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_persian_ner_shot.txt",
            fa=PROMPTS_PATH / "fa_persian_ner_shot.txt",
        ),
        prompt_variables_mapping={"input": "input"},
        target_variables_mapping={"output": "output"},
    )

    scorer: Scorer = ner_exact_match

    def score_matches(self, matches: TaskMatchGroup) -> TaskMatchGroup:
        clean_text = lambda text: text.replace("\u200c", " ").lower()

        def extract_ner(completion):
            completion = clean_text(completion)
            start_idx = completion.find("[(")
            if start_idx == -1:
                return ""
            end_idx = completion.find(")]", start_idx)
            if end_idx == -1:
                return ""
            completion = completion[
                max(start_idx - 2, 0) : min(end_idx + 2, len(completion))
            ]
            return completion

        matches.format_completions(extract_ner)
        matches.format_targets(clean_text)

        return super().score_matches(matches)
