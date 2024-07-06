import os
import re
from pathlib import Path

import datasets

from parsbench.tasks.base import (
    ConstantPromptVariable,
    HuggingFaceDataLoader,
    LazyLoadTemplates,
    PromptTemplate,
    Task,
    TaskCategory,
    TaskMatchGroup,
)

PROMPTS_PATH = Path(__file__).parent / "prompts"

CATEGORIES_MAPPING = {
    "جغرافیا دوره دوم متوسطه": "Geography USS",
    "علوم تجربی دوره اول ابتدایی": "Natural Sciences LPS",
    "تعلیم و تربیت اسلامی دوره اول متوسطه": "Theology LSS",
    "زبان و ادبیات فارسی دوره دوم ابتدایی": "Persian Literature LPS",
    "مطالعات اجتماعی دوره اول ابتدایی": "Social Studies LSS",
    "هوش کلامی و ادبی دوره دوم ابتدایی": "Verbal and Linguistic Intelligence UPS",
    "فلسفه دوره دوم متوسطه": "Philosophy USS",
    "شیمی دوره دوم متوسطه": "Chemistry USS",
    "ریاضیات گسسته دوره دوم متوسطه": "Discrete Mathematics USS",
    "تاریخ دوره دوم متوسطه": "History USS",
    "اقتصاد دوره دوم متوسطه": "Economy USS",
    "تعلیم و تربیت اسلامی دوره اول ابتدایی": "Theology USS",
    "هندسه دوره دوم متوسطه": "Geometry USS",
    "زبان و ادبیات فارسی دوره اول متوسطه": "Persian Literature LSS",
    "آمار و احتمال دوره دوم متوسطه": "Probability and Statistics USS",
    "مطالعات اجتماعی دوره دوم ابتدایی": "Social Studies LPS",
    "مطالعات اجتماعی دوره اول متوسطه": "Social Studies UPS",
    "علوم تجربی دوره دوم ابتدایی": "Natural Sciences UPS",
    "ریاضی دوره اول متوسطه": "Mathematics LPS",
    "تعلیم و تربیت اسلامی دوره دوم ابتدایی": "Theology LPS",
    "سرعت و دقت دوره دوم ابتدایی": "Speed and Accuracy UPS",
    "ریاضی دوره دوم متوسطه": "Mathematics LSS",
    "زبان و ادبیات فارسی دوره دوم متوسطه": "Persian Literature USS",
    "دین و زندگی دوره دوم متوسطه": "Theology UPS",
    "ریاضی و آمار دوره دوم متوسطه": "Mathematics and Statistics USS",
    "استعداد تحلیلی دوره اول متوسطه": "Analytical Talent LSS",
    "روانشناسی دوره دوم متوسطه": "Psychology USS",
    "زمین شناسی دوره دوم متوسطه": "Geology USS",
    "جامعه شناسی دوره دوم متوسطه": "Sociology USS",
    "ریاضی دوره دوم ابتدایی": "Mathematics USS",
    "زیست شناسی دوره دوم متوسطه": "Biology USS",
    "هوش ریاضی و منطقی دوره دوم ابتدایی": "Mathematical and Logical Intelligence UPS",
    "زبان و ادبیات فارسی دوره اول ابتدایی": "Persian Literature UPS",
    "ریاضی دوره اول ابتدایی": "Mathematics UPS",
    "حسابان دوره دوم متوسطه": "Calculus USS",
    "علوم تجربی دوره اول متوسطه": "Natural Sciences LSS",
    "فیزیک دوره دوم متوسطه": "Physics USS",
    "منطق دوره دوم متوسطه": "Logic USS",
}


class _MultipleChoiceDataLoader(HuggingFaceDataLoader):
    def load(self) -> list[dict]:
        print("Preparing Persian MMLU dataset:")

        token = os.environ.get("HF_TOKEN", None)
        if not token:
            raise Exception(
                "Environment variable HF_TOKEN (HuggingFace Token) not found. "
                "You should set it for downloading the PersianMMLU dataset."
            )

        ds = datasets.load_dataset(
            self.data_path,
            split=self.split,
            token=token,
        )

        ds = ds.select_columns(
            column_names=[
                "ID",
                "Question Body",
                "Choice 1",
                "Choice 2",
                "Choice 3",
                "Choice 4",
                "Key",
                "final_category_fa",
            ],
        )

        new_features = ds.features.copy()
        new_features["Key"] = datasets.Value("string")
        ds = ds.cast(new_features)

        merge_choices = lambda d: [
            d["Choice 1"],
            d["Choice 2"],
            d["Choice 3"],
            d["Choice 4"],
        ]
        ds = ds.map(
            lambda data: {
                "choices": merge_choices(data),
                "category": CATEGORIES_MAPPING[data["final_category_fa"]],
            },
            remove_columns=[
                "Choice 1",
                "Choice 2",
                "Choice 3",
                "Choice 4",
                "final_category_fa",
            ],
        )

        return ds.to_list()


class _MultipleChoicePromptTemplate(PromptTemplate):
    def get_prompt_variables(self, data: dict) -> dict:
        mapped_data = {}
        for pk, dk in self.prompt_variables_mapping.items():
            if isinstance(dk, ConstantPromptVariable):
                mapped_data[pk] = dk.value
            else:
                if dk not in data:
                    raise ValueError(f"Key {dk} not in data.")
                if dk == "choices":
                    mapped_data[pk] = "\n".join(
                        map(lambda p: f"{p[0]}. {p[1]}", zip("1234", data[dk]))
                    )
                else:
                    mapped_data[pk] = data[dk]
        return mapped_data


class PersianMMLU(Task):
    task_name: str = "Persian MMLU"
    task_category: TaskCategory = TaskCategory.KNOWLEDGE

    data_loader: HuggingFaceDataLoader = _MultipleChoiceDataLoader(
        data_path="raia-center/khayyam-challenge",
        split="train",
    )
    data_target_key: str = "Key"

    sub_task_key: str = "category"
    sub_tasks: list[str] = list(CATEGORIES_MAPPING.values())

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
            "question": "Question Body",
            "candidates": "choices",
        },
        target_variables_mapping={"answer": "Key"},
    )

    def score_matches(self, matches: TaskMatchGroup) -> TaskMatchGroup:
        def _extract_number(completion) -> str:
            if m := re.match(r"\d{1}", completion):
                return m.group()
            return "-1"

        matches.format_completions(_extract_number)

        return super().score_matches(matches)
