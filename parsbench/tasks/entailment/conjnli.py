from pathlib import Path

from sklearn.metrics import f1_score

from parsbench.tasks.base import (
    JSONLineDataLoader,
    LazyLoadTemplates,
    PromptTemplate,
    Task,
    TaskCategory,
    TaskMatchGroup,
)

PROMPTS_PATH = Path(__file__).parent / "prompts"
ACCEPTED_TARGETS = ["c", "e", "n"]
DATA_URL = "https://raw.githubusercontent.com/Ipouyall/Benchmarking_ChatGPT_for_Persian/main/Benchmark/Entailment(conjnli)/data.jsonl"


class ConjNLIEntailment(Task):
    task_name: str = "ConjNLI Entailment"
    task_category: TaskCategory = TaskCategory.REASONING

    data_loader: JSONLineDataLoader = JSONLineDataLoader(data_path=DATA_URL)
    data_target_key: str = "target"

    prompt_template: PromptTemplate = PromptTemplate(
        language_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_conjnli_entailment.txt",
            fa=PROMPTS_PATH / "fa_conjnli_entailment.txt",
        ),
        prompt_shot_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_conjnli_entailment_shot.txt",
            fa=PROMPTS_PATH / "fa_conjnli_entailment_shot.txt",
        ),
        prompt_variables_mapping={
            "premise": "Fa_premise",
            "hypothesis": "Fa_hypothesis",
        },
        target_variables_mapping={"target": "target"},
    )

    def score_matches(self, matches: TaskMatchGroup) -> TaskMatchGroup:
        matches.format_completions(lambda c: c.strip(" ").strip("'").lower())
        return super().score_matches(matches)

    def get_overall_score(self, matches: TaskMatchGroup) -> float:
        return f1_score(
            y_true=matches.targets,
            y_pred=matches.completions,
            labels=ACCEPTED_TARGETS,
            average="macro",
            zero_division=0,
        )
