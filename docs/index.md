# ParsBench

<div align="center">
    <a href="https://github.com/shahriarshm/parsbench">
        <img src="https://raw.githubusercontent.com/shahriarshm/parsbench/main/docs/imgs/banner-black.png" alt="Beanie" width="480" height="240">
    </a>
    <br>
    <a href="https://shahriarshm.github.io/parsbench/">
        <img src="https://shields.io/badge/-docs-blue" alt="docs">
    </a>
    <a href="https://pypi.python.org/pypi/parsbench">
        <img src="https://img.shields.io/pypi/v/parsbench.svg" alt="pypi">
    </a>
    <a href="https://huggingface.co/ParsBench">
        <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-md-dark.svg" alt="huggingface">
    </a>
</div>

## Overview

ParsBench provides toolkits for benchmarking Large Language Models (LLMs) based on the Persian language. It includes various tasks for evaluating LLMs on different topics, benchmarking tools to compare multiple models and rank them, and an easy, fully customizable API for developers to create custom models, tasks, scores, and benchmarks.

## Key Features

- **Variety of Tasks**: Evaluate LLMs across various topics.
- **Benchmarking Tools**: Compare and rank multiple models.
- **Customizable API**: Create custom models, tasks, scores, and benchmarks with ease.

## Motivation

I was trying to fine-tune an open-source LLM for the Persian language. I needed some evaluation to test the performance and utility of my LLM. It leads me to research and find [this paper](https://arxiv.org/abs/2404.02403). It's great work that they prepared some datasets and evaluation methods to test on ChatGPT. They even shared their code in this [repository](https://github.com/Ipouyall/Benchmarking_ChatGPT_for_Persian).

So, I thought that I should build a handy framework that includes various tasks and datasets for evaluating LLMs based on the Persian language. I used some parts of their work (Datasets, Metrics, Basic prompt templates) in this library.

## Example Notebooks

- Benchmark [Aya](https://huggingface.co/CohereForAI) models: [![aya](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aPayB9AaheDxT7zS4A_4SAMH3a7mIDFX?usp=sharing)
- Benchmark [Ava](https://huggingface.co/MehdiHosseiniMoghadam) models: [![ava](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1ToJ8gTQz1ifU70EBAM7fZG2LIOY4zAp0/view?usp=sharing)
- Benchmark [Dorna](https://huggingface.co/PartAI) models: [![dorna](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1f64d0GnmcQIZ-tlN8cg49pPdiwlVlWvi/view?usp=sharing)
- Benchmark [MaralGPT](https://huggingface.co/MaralGPT) models: [![maralgpt](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1ZfjxPa4CfAZdQgtPaEt3nnX180A825ZF/view?usp=sharing)

## Contributing

Contributions are welcome! Please refer to the [contribution guidelines](contribution.md) for more information on how to contribute.

## License

ParsBench is distributed under the Apache-2.0 license.

## Contact Information

For support or questions, please contact: [shahriarshm81@gmail.com](mailto:shahriarshm81@gmail.com)
Feel free to let me know if there are any additional details or changes you'd like to make!
