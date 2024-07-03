import re
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
ACCEPTED_TARGETS = ["-3", "-2", "-1", "0", "1", "2", "3"]


class ParsiNLUSentimentAnalysis(Task):
    task_name: str = "Sentiment Analysis"
    task_category: TaskCategory = TaskCategory.CLASSIC

    data_loader: HuggingFaceDataLoader = HuggingFaceDataLoader(
        data_path="persiannlp/parsinlu_sentiment",
        split="test_food",
    ).with_filter(lambda data: data["aspect"] == "کلی")

    prompt_template: PromptTemplate = PromptTemplate(
        language_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_parsinlu_sentiment.txt",
            fa=PROMPTS_PATH / "fa_parsinlu_sentiment.txt",
        ),
        prompt_shot_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_parsinlu_sentiment_shot.txt",
            fa=PROMPTS_PATH / "fa_parsinlu_sentiment_shot.txt",
        ),
        prompt_variables_mapping={"review": "review"},
        target_variables_mapping={"label": "label"},
    )

    def score_matches(self, matches: TaskMatchGroup) -> TaskMatchGroup:
        def _format_completion(completion):
            completion = completion.replace("'", "")
            match = re.match(r"(-?\d)", completion)
            if not match or match[0] not in ACCEPTED_TARGETS:
                return "-4"
            return match[0]

        matches.format_completions(_format_completion)

        return super().score_matches(matches)

    def get_overall_score(self, matches: TaskMatchGroup) -> float:
        return f1_score(
            y_true=matches.targets,
            y_pred=matches.completions,
            labels=ACCEPTED_TARGETS,
            average="macro",
            zero_division=0,
        )
