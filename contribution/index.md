# Contribution

Thank you for considering contributing to ParsBench! Contributions of code, tasks, datasets, and benchmark results all help. Here are the guidelines.

## Ways to contribute

### 1. Fix bugs

If you find a bug in ParsBench:

- Report it in the [issue tracker](https://github.com/ParsBench/ParsBench/issues) with steps to reproduce, the expected behavior, and the actual behavior.
- Or submit a pull request with a fix. Please include relevant tests and documentation updates where applicable.

### 2. Add features

If you have an idea for a new feature:

- Propose it in the [issue tracker](https://github.com/ParsBench/ParsBench/issues) first, describing its purpose and impact, so we can discuss the design before you invest time in it.
- Then submit a PR with the implementation, tests, and documentation with usage examples.

### 3. Add new tasks with new datasets

New Persian evaluation tasks expand what ParsBench can measure:

- Propose the task in the [issue tracker](https://github.com/ParsBench/ParsBench/issues), including its objective, its relevance to Persian language processing, and the dataset's source, format, and any preprocessing.
- Submit a PR that adds the task, integrates the dataset, includes tests, and documents how to use it. The [advanced tutorial](https://parsbench.github.io/ParsBench/tutorial/advanced/index.md) shows the anatomy of a task.

### 4. Run benchmarks on new models

Benchmark results on state-of-the-art or fine-tuned open-weight models are contributions too:

- Run the benchmarks with ParsBench and share the results in the [issue tracker](https://github.com/ParsBench/ParsBench/issues), including details about the model and any fine-tuning, the saved matches, and your observations.

## Development setup

ParsBench uses [Poetry](https://python-poetry.org/) and requires Python 3.12+:

```
git clone https://github.com/<your-username>/ParsBench.git
cd ParsBench
poetry install

poetry run pytest              # run the test suite
poetry run mkdocs serve        # preview the documentation locally
```

## Contribution process

1. Fork the [ParsBench repository](https://github.com/ParsBench/ParsBench) to your GitHub account and clone it.
1. Create a branch for your change:

```
git checkout -b feature/your-feature-name
```

1. Make your changes, commit with a meaningful message, and push:

```
git commit -m "Description of your changes"
git push origin feature/your-feature-name
```

1. Open a pull request against the main repository with a description of what changed and why.

## Code style and testing

- Follow the existing code style and conventions (the project uses `black` and `isort`).
- Include tests for your changes and make sure the suite passes before submitting.
- Update the documentation when behavior changes.

## Getting help

If you have questions or get stuck, open an issue in the [issue tracker](https://github.com/ParsBench/ParsBench/issues).
