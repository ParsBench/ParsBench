# Generating goldens

Writing the first thirty goldens by hand is the boring part of adopting any
eval tool, and it's where most teams stall. `GoldenGenerator` bootstraps a
suite from documentation you already have: your product FAQ, knowledge base,
or policy pages.

```python
from parsbench.appeval import AppEvaluator, GoldenGenerator

goldens = GoldenGenerator(model="gpt-4.1-mini", adversarial=True).generate("kb/", n=30)
result = AppEvaluator(goldens).evaluate(my_bot)
```

`generate()` accepts a file path, a glob pattern, a directory, or a list of
those. Text-like files only (`.txt`, `.md`, `.rst`, `.html`, `.json`). It
chunks the documents, asks the generator model for questions a real user
would ask about each chunk, and returns ready `Golden` objects.

Three things make the output more than generic QA pairs:

- Questions rotate through registers, formal and colloquial Persian by
  default. Pass `registers=[...]` to change the rotation.
- `adversarial=True` mixes digit scripts, Jalali dates, and Finglish into
  some questions, the same traps the [user simulator](simulation.md) plays.
- Each generated golden carries its source chunk as `context=`, so the
  [faithfulness check](goldens.md) runs automatically: a judge verifies the
  bot's answer is supported by the document the question came from.

The generator model resolves like the judge does: `model=` argument, then the
`PARSBENCH_GENERATOR` env var, then `PARSBENCH_JUDGE`. Unlike judge checks,
generation raises without a model, since there is nothing useful to do
without one.

## Review before you trust

Generated goldens are a starting point, not ground truth. Skim them, delete
the bad ones, and tighten the good ones with explicit `contains=` or `tools=`
expectations where you know the right answer. A practical loop:

```python
import json
from dataclasses import asdict

goldens = GoldenGenerator(model="gpt-4.1-mini").generate("kb/", n=50)
with open("goldens.json", "w") as f:
    json.dump([asdict(g) for g in goldens], f, ensure_ascii=False, indent=2, default=str)
```

Edit the JSON by hand, then load it back. `AppEvaluator` accepts dicts
directly:

```python
with open("goldens.json") as f:
    result = AppEvaluator(goldens=json.load(f)).evaluate(my_bot)
```

See
[`examples/industry/rag_faq_support.py`](https://github.com/ParsBench/ParsBench/blob/main/examples/industry/rag_faq_support.py)
for a full knowledge-base RAG evaluation built this way.
