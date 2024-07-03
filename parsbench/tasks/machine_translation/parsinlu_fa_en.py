from pathlib import Path

from parsbench import scores
from parsbench.scores.base import Scorer
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


class ParsiNLUMachineTranslationFaEn(Task):
    task_name: str = "ParsiNLU Machine Translation Fa-En"
    task_category: TaskCategory = TaskCategory.CLASSIC

    data_loader: HuggingFaceDataLoader = HuggingFaceDataLoader(
        data_path="persiannlp/parsinlu_translation_fa_en",
        split="validation",
    )
    data_target_key: str = "targets"

    prompt_template: PromptTemplate = PromptTemplate(
        language_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_parsinlu_translation.txt",
            fa=PROMPTS_PATH / "fa_parsinlu_translation.txt",
        ),
        prompt_shot_templates=LazyLoadTemplates(
            en=PROMPTS_PATH / "en_parsinlu_translation_shot.txt",
            fa=PROMPTS_PATH / "fa_parsinlu_translation_shot.txt",
        ),
        prompt_variables_mapping={
            "input": "source",
            "input_language": ConstantPromptVariable("Persian"),
            "output_language": ConstantPromptVariable("English"),
        },
        target_variables_mapping={"output": "targets"},
    )

    scorer: Scorer = scores.english_sentence_bleu

    def score_matches(self, matches: TaskMatchGroup) -> TaskMatchGroup:
        matches.format_targets(lambda ts: ts[0])
        return super().score_matches(matches)
