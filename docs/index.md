# ParsBench

[![Beanie](https://raw.githubusercontent.com/shahriarshm/parsbench/main/docs/imgs/banner-black.png)](https://github.com/shahriarshm/parsbench)

[![docs](https://shields.io/badge/-docs-blue)](https://shahriarshm.github.io/parsbench/)
[![pypi](https://img.shields.io/pypi/v/parsbench.svg)](https://pypi.python.org/pypi/parsbench)

## Overview

ParsBench provides toolkits for benchmarking Large Language Models (LLMs) based on the Persian language. It includes various tasks for evaluating LLMs on different topics, benchmarking tools to compare multiple models and rank them, and an easy, fully customizable API for developers to create custom models, tasks, scores, and benchmarks.

## Key Features

- **Variety of Tasks**: Evaluate LLMs across various topics.
- **Benchmarking Tools**: Compare and rank multiple models.
- **Customizable API**: Create custom models, tasks, scores, and benchmarks with ease.

## Motivation

I was trying to fine-tune an open-source LLM for the Persian language. I needed some evaluation to test the performance and utility of my LLM. It leads me to research and find [this paper](https://arxiv.org/abs/2404.02403). It's great work that they prepared some datasets and evaluation methods to test on ChatGPT. They even shared their code in this [repository](https://github.com/Ipouyall/Benchmarking_ChatGPT_for_Persian).

So, I thought that I should build a handy framework that includes various tasks and datasets for evaluating LLMs based on the Persian language. I used some parts of their work (Datasets, Metrics, Basic prompt templates) in this library.

## Contributing

Contributions are welcome! Please refer to the [contribution guidelines](contribution.md) for more information on how to contribute.

## License

ParsBench is distributed under the Apache-2.0 license.

## Contact Information

For support or questions, please contact: [shahriarshm81@gmail.com](mailto:shahriarshm81@gmail.com)
Feel free to let me know if there are any additional details or changes you'd like to make!
