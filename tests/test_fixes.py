"""Regression tests for the code-review fixes.

Runnable with pytest or directly: `python tests/test_fixes.py`.
"""

from parsbench.benchmarks.benchmark_result import (
    BenchmarkResult,
    ModelBenchmarkResult,
    merge_benchmark_results,
)
from parsbench.models.base import Model
from parsbench.scores.summarization import _persian_scorer, persian_rouge
from parsbench.tasks.base import Task, TaskCategory, TaskMatch, TaskMatchGroup
from parsbench.tasks.ner.persian_ner import ner_exact_match


class _EchoModel(Model):
    """Returns the prompt back, so completion == target in the mini task."""

    support_concurrency = False

    @property
    def model_name(self) -> str:
        return "echo"

    def get_prompt_completion(self, prompt: str) -> str:
        return prompt

    def prompt_formatter(self, prompt: str) -> str:
        return prompt


class _MiniTask(Task):
    task_name = "Mini"
    task_category = TaskCategory.CLASSIC

    def generate_matches(
        self, prompt_lang, n_shots=0, n_first=None, sub_task=None
    ) -> TaskMatchGroup:
        matches = [TaskMatch(id=i + 1, prompt=str(i), target=str(i)) for i in range(5)]
        return TaskMatchGroup(n_shots=n_shots, matches=matches)


def _mbr(name: str) -> ModelBenchmarkResult:
    return ModelBenchmarkResult(model_name=name, evaluation_results=[])


def test_evaluate_multi_shot_builds_single_result():
    # Bug: EvaluationResult was built inside the shots loop over *all* groups,
    # crashing (AssertionError) on multi-shot. Should yield one result per sub_task.
    results = _MiniTask().evaluate(
        model=_EchoModel(),
        prompt_shots=[0, 1],
        n_first=None,
    )
    assert len(results) == 1
    result = results[0]
    assert [psr.n_shots for psr in result.prompt_shot_results] == [0, 1]
    assert all(psr.score == 1.0 for psr in result.prompt_shot_results)


def test_merge_dedup_keeps_first_drops_later_duplicates():
    merged = merge_benchmark_results(
        [BenchmarkResult([_mbr("A"), _mbr("B")]), BenchmarkResult([_mbr("A")])],
        sort=False,
    )
    assert [mb.model_name for mb in merged.model_benchmarks] == ["A", "B"]


def test_merge_dedup_keep_duplicates_flag():
    merged = merge_benchmark_results(
        [BenchmarkResult([_mbr("A")]), BenchmarkResult([_mbr("A")])],
        sort=False,
        keep_duplicates=True,
    )
    assert [mb.model_name for mb in merged.model_benchmarks] == ["A", "A"]


def test_persian_rouge_scorer_is_cached():
    # Optimization: scorer must be built once and reused, not per completion.
    assert _persian_scorer() is _persian_scorer()
    assert persian_rouge.measure("سلام دنیا", "سلام دنیا") == 1.0


def test_ner_scorer_is_safe_and_correct():
    # No eval() execution on model output, and no crash on empty parse.
    assert ner_exact_match.measure("__import__('os').getcwd()", "[('a', 'o')]") == 0
    assert ner_exact_match.measure("[]", "[('a', 'o')]") == 0
    assert ner_exact_match.measure("[('tehran', 'loc')]", "[('tehran', 'loc')]") == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok - {name}")
    print("all passed")
