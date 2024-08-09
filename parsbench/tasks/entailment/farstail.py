from pathlib import Path

from sklearn.metrics import f1_score

from parsbench.tasks.base import (
    CSVDataLoader,
    LazyLoadTemplates,
    PromptTemplate,
    Task,
    TaskCategory,
    TaskMatchGroup,
)

PROMPTS_PATH = Path(__file__).parent / "prompts"
ACCEPTED_TARGETS = ["c", "e", "n"]
DATA_URL = (
    "https://raw.githubusercontent.com/dml-qom/FarsTail/master/data/Test-word.csv"
)


class FarsTailEntailment(Task):
    task_name: str = "FarsTail Entailment"
    task_category: TaskCategory = TaskCategory.REASONING

    data_loader: CSVDataLoader = CSVDataLoader(
        data_path=DATA_URL, csv_arguments={"delimiter": "\t", "quotechar": '"'}
    )
    data_target_key: str = "label"

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
            "premise": "premise",
            "hypothesis": "hypothesis",
        },
        target_variables_mapping={"target": "label"},
    )

    def score_matches(self, matches: TaskMatchGroup) -> TaskMatchGroup:
        def _find_letter(completion: str) -> str:
            completion = completion.lower()
            for letter in ACCEPTED_TARGETS:
                if letter in completion:
                    return letter
            return ""

        matches.format_completions(_find_letter)
        return super().score_matches(matches)

    def get_overall_score(self, matches: TaskMatchGroup) -> float:
        return f1_score(
            y_true=matches.targets,
            y_pred=matches.completions,
            labels=ACCEPTED_TARGETS,
            average="macro",
            zero_division=0,
        )
