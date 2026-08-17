# Changelog

## 0.3.0 - 2026-08-17

### Added

- App evaluation (`parsbench.appeval`), in the same class-based style as
  tasks and benchmarks: `AppEvaluator(goldens).evaluate(app)` scores your own
  Persian AI app with `Golden` expectations — tool calls,
  contains/not_contains, structured format, budgets, custom checks, and
  Persian-prompted judge checks (correctness, faithfulness, refusal) — and
  `score_traces()` evaluates pre-captured traces instead, with `pass^k` for
  flaky agents. Results are plain dataclasses (`AppEvaluationResult`) with
  the familiar `to_dict`/`to_pandas`/`save` conveniences.
- Persian-aware matching layer: Arabic/Persian codepoint and digit-script
  unification, ZWNJ/spacing tolerance, rial/toman + هزار/میلیون amounts
  (including compound «۲ میلیون و ۵۰۰ هزار» and the ۲٫۵ / ۲/۵ decimal
  forms), and Jalali/Gregorian date equivalence — applied symmetrically to
  `contains`, `not_contains`, and tool-argument comparison.
- Multi-turn simulation (`SimulationEvaluator`) with an Iranian-user
  simulator (taarof, Finglish, toman/rial confusion, Jalali dates, …), test
  generation from your docs (`GoldenGenerator`), and judge calibration
  against human labels (`JudgeCalibrator`).
- Framework integrations: OpenAI Agents SDK, LangGraph/LangChain,
  Pydantic AI, Agno, Google ADK, and any OTel-instrumented app via
  `TraceCollector`; Langfuse score export via `result.to_langfuse()`.
- `parsbench view`: a local, ParsBench-themed viewer over recorded runs —
  live progress, failure-first run pages, trace detail, RTL simulation
  replay, run-vs-run diffs, dark/light themes, JSON/CSV/Markdown export,
  and optional score charts. Zero new dependencies, no build step.
- Evaluations now record into a project-local `.parsbench/` store by
  default (full traces included; opt out with `record=False` or
  `PARSBENCH_NO_RECORD=1`).
- `parsbench test` CLI (pytest wrapper; install with `parsbench[test]`) and
  runnable examples for every supported framework under `examples/`, plus
  industry scenarios (banking, e-commerce, healthcare, telecom simulation,
  knowledge-base RAG) under `examples/industry/` — the offline ones run in CI.
- Production hardening: a crash inside the evaluated app is reported as a
  failing `app_error` check instead of aborting the run, async apps work
  inside running event loops (notebooks, servers), evaluation and calibration
  accept `prefer_concurrency=`/`n_workers=`, and the built-in judge client
  retries transient failures with a bounded per-call timeout
  (`PARSBENCH_MAX_RETRIES` / `PARSBENCH_TIMEOUT`). The package ships
  `py.typed`, the app-eval surface is mypy-clean, and CI runs the suite on
  Python 3.12/3.13.

### Changed

- Simulation: hitting the turn cap no longer fails a conversation the goal
  judge scored as successful — a chatty user simulator that never emits the
  stop token was punishing the app for the simulator's behavior.

## 0.2.0 - 2026-07-15

### Changed

- Support current library versions: transformers 5.x, datasets 5.x, openai 2.x, anthropic, and numpy 2.
- **Breaking:** drop Python 3.10/3.11 support; Python >= 3.12 is now required (needed by hazm >= 0.11 and numpy 2).
- Bump hazm to 0.12, drop the unused `scipy` pin, and declare the `numpy`/`pandas`/`requests`/`tqdm`/`nltk` dependencies that were previously only installed transitively.

### Fixed

- Fix evaluation and merge correctness bugs, and cache the summarization/NER scorers.
- Use cleaned completions when scoring matches.
- Replace the undeclared `pytz` dependency with the standard library (`pandas` 3 no longer ships it).
- Remove format targets in the Persian Math task.
- Import the optional `math_equivalence` package lazily so `import parsbench` no longer fails when it is not installed.

### Added

- Add `show_bar_plot` to `BenchmarkResult`.
- Add a mechanism to skip evaluation results on error.

## 0.1.7 - 2024-08-15

### Fixed

- Fix typos in prompt templates.
- Fix error on using formatted targets while scoring matches.
- Improve `from_matches_files` function speed in BenchmarkResult.
- Fix returning list in AnthropicModel completion function.

### Added

- Add `formatted_completion` field to task matches.
- Update completion formatters in tasks.
- Add re-score option to `from_matches_files` in BenchmarkResult.
- Add max retries exceeded error in API-based models.
- Add snapshot functionality to save matches on error.
- Add leaderboard builder function.
- Add build from file functions to the BenchmarkResult class.

## 0.1.6 - 2024-07-25

### Fixed

- Fix misspell in ParsiNLUMultipleChoice task name.
- Fix wrong target key in the XLSummary.
- Add org prefix to the sentiment analysis task.
- Fix FarsTailEntailment prompt target key.

### Added

- Add `attention_mask` to the transformer model `generate` function.

## 0.1.5 - 2024-07-18

### Added

- Add FarsTail entailment task.
- Add Persian News Summary task.
- Add XL-Sum task.
- Add ParsiNLUBenchmark. It is a sub class of CustomBenchmark with hard-coded ParsiNLU tasks.

## 0.1.4 - 2024-07-12

### Fixed

- Fix `load_all_tasks` returning empty list.

### Added

- Add Anthropic model interface.
- Add retry on rate limit to API-based models.
- Add `skip_existing_matches` to the task evaluate function. It skips matches that are already generated and scored.

## 0.1.3 - 2024-07-06

### Fixed

- Fix `model_name` property in PreTrainedTransformerModel.

### Added

- Add Persian MMLU (Khayyam Challenge) task.
- Add `select_sub_tasks` to the task class.

## 0.1.2 - 2024-07-06

### Fixed

- Fix misspells and typos.
- Use`name_or_path` parameter as the `model_name` in PreTrainedTransformerModel.

### Changed

- Update sentiment analysis task prompt template.

### Added

- Add `completion_formatter` to the model interfaces.

## 0.1.1 - 2024-07-05

### Added

- Add support for Python >= 3.10
- Add `prefer_concurrency` to the benchmark, task and models.

## 0.1.0 - 2024-07-03

ParsBench got alive!
