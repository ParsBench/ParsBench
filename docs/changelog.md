# Changelog

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
