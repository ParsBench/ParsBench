# Changelog

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
