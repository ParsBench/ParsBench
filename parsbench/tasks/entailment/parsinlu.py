from pathlib import Path

from sklearn.metrics import f1_score

from parsbench.tasks.base import (
    HuggingFaceDataLoader,
    LazyLoadTemplates,
    PromptTemplate,
    Task,
    TaskCategory,
    TaskMatchGroup,
)

PROMPTS_PATH = Path(__file__).parent / "prompts"
ACCEPTED_TARGETS = ["c", "e", "n"]


class ParsiNLUEntailment(Task):
    task_name: str = "ParsiNLU Entailment"
    task_category: TaskCategory = TaskCategory.REASONING

    data_loader: HuggingFaceDataLoader = HuggingFaceDataLoader(
        data_path="persiannlp/parsinlu_entailment",
        split="test",
    )

    prompt_template: PromptTemplate = PromptTemplate(
        language_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_parsinlu_entailment.txt",
            fa=PROMPTS_PATH / "fa_parsinlu_entailment.txt",
        ),
        prompt_shot_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_parsinlu_entailment_shot.txt",
            fa=PROMPTS_PATH / "fa_parsinlu_entailment_shot.txt",
        ),
        prompt_variables_mapping={"premise": "sent1", "hypothesis": "sent2"},
        target_variables_mapping={"label": "label"},
    )

    def score_matches(self, matches: TaskMatchGroup) -> TaskMatchGroup:
        completion_mapper = {
            "contradiction": "c",
            "entailment": "e",
            "neutral": "n",
            "تناقض": "c",
            "تناظر": "e",
            "ناشناخته": "n",
        }
        matches.format_completions(
            lambda c: completion_mapper.get(c.strip().strip("'").lower(), "")
        )
        return super().score_matches(matches)

    def get_overall_score(cls, matches: TaskMatchGroup) -> float:
        return f1_score(
            y_true=matches.targets,
            y_pred=matches.completions,
            labels=ACCEPTED_TARGETS,
            average="macro",
            zero_division=0,
        )
