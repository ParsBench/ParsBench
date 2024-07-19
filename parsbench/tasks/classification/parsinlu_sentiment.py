from pathlib import Path

import datasets
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

ACCEPTED_TARGETS = ["POSITIVE", "NEGATIVE", "NEUTRAL", "OTHER"]
DATASET_TARGETS_MAPPING = {
    "-2": "NEGATIVE",
    "-1": "NEGATIVE",
    "0": "NEUTRAL",
    "1": "POSITIVE",
    "2": "POSITIVE",
    "3": "OTHER",
}


class _SentimentDataLoader(HuggingFaceDataLoader):
    def load(self) -> list[dict]:
        ds = datasets.load_dataset(self.data_path, split=self.split)
        ds = ds.filter(lambda data: data["aspect"] == "کلی")

        def _label_mapper(data):
            data["label"] = DATASET_TARGETS_MAPPING[data["label"]]
            return data

        ds = ds.map(_label_mapper)

        return ds.to_list()


class ParsiNLUSentimentAnalysis(Task):
    task_name: str = "ParsiNLU Sentiment Analysis"
    task_category: TaskCategory = TaskCategory.CLASSIC

    data_loader: HuggingFaceDataLoader = _SentimentDataLoader(
        data_path="persiannlp/parsinlu_sentiment",
        split="test_food",
    )

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
            for target in ACCEPTED_TARGETS:
                if target in completion:
                    return target
            return ""

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
