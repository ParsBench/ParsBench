from pathlib import Path

from parsbench import scores
from parsbench.scores.base import Scorer
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


class XLSummary(Task):
    task_name: str = "XLSummary"
    task_category: TaskCategory = TaskCategory.LANGUAGE

    data_loader: HuggingFaceDataLoader = HuggingFaceDataLoader(
        data_path="csebuetnlp/xlsum",
        split="test",
        name="persian",
    )
    data_target_key: str = "summary"

    prompt_template: PromptTemplate = PromptTemplate(
        language_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_pnsummary.txt",
            fa=PROMPTS_PATH / "fa_pnsummary.txt",
        ),
        prompt_shot_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_pnsummary_shot.txt",
            fa=PROMPTS_PATH / "fa_pnsummary_shot.txt",
        ),
        prompt_variables_mapping={"article": "text"},
        target_variables_mapping={"summary": "summary"},
    )

    scorer: Scorer = scores.persian_rouge

    def score_matches(self, matches: TaskMatchGroup) -> TaskMatchGroup:
        matches.format_completions(lambda c: c.strip("`").strip("'").lower())
        return super().score_matches(matches)
