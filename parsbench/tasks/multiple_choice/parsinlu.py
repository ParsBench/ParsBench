import re
from pathlib import Path

from parsbench.tasks.base import (
    ConstantPromptVariable,
    JSONLineDataLoader,
    LazyLoadTemplates,
    PromptTemplate,
    Task,
    TaskCategory,
    TaskMatchGroup,
)

PROMPTS_PATH = Path(__file__).parent / "prompts"
DATA_URL = "https://raw.githubusercontent.com/persiannlp/parsinlu/master/data/multiple-choice/test.jsonl"


class _MultipleChoicePromptTemplate(PromptTemplate):
    def get_prompt_variables(self, data: dict) -> dict:
        mapped_data = {}
        for pk, dk in self.prompt_variables_mapping.items():
            if isinstance(dk, ConstantPromptVariable):
                mapped_data[pk] = dk.value
            else:
                if dk not in data:
                    raise ValueError(f"Key {dk} not in data.")
                if dk == "candidates":
                    mapped_data[pk] = "\n".join(
                        map(lambda p: f"{p[0]}. {p[1]}", zip("1234", data[dk]))
                    )
                else:
                    mapped_data[pk] = data[dk]
        return mapped_data


class ParsiNLUMultipleChoice(Task):
    task_name: str = "PersiNLU Multiple Choice"
    task_category: TaskCategory = TaskCategory.KNOWLADGE

    data_loader: JSONLineDataLoader = JSONLineDataLoader(data_path=DATA_URL)
    data_target_key: str = "answer"

    sub_task_key: str = "category"
    sub_tasks: list[str] = ["math_and_logic", "common_knowledge", "literature"]

    prompt_template: PromptTemplate = _MultipleChoicePromptTemplate(
        language_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_parsinlu_multiple_choice.txt",
            fa=PROMPTS_PATH / "fa_parsinlu_multiple_choice.txt",
        ),
        prompt_shot_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_parsinlu_multiple_choice_shot.txt",
            fa=PROMPTS_PATH / "fa_parsinlu_multiple_choice_shot.txt",
        ),
        prompt_variables_mapping={
            "question": "question",
            "candidates": "candidates",
        },
        target_variables_mapping={"answer": "answer"},
    )

    def score_matches(self, matches: TaskMatchGroup) -> TaskMatchGroup:
        def _extract_number(completion) -> str:
            if m := re.match(r"\d{1}", completion):
                return m.group()
            return "-1"

        matches.format_completions(_extract_number)

        return super().score_matches(matches)
