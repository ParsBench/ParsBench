"""GoldenGenerator: generate Persian goldens from the user's own documents.

Chunk + LLM-JSON generation; upgrade to knowledge-graph synthesis
(RAGAS-style) if single-chunk questions prove too shallow.
"""

import glob
import json
import re
from pathlib import Path
from typing import Any

from .golden import Golden
from .judge import resolve_model

TESTGEN_FA = (
    "متن زیر بخشی از مستندات یک محصول است:\n\n---\n{chunk}\n---\n\n"
    "بر اساس فقط همین متن، {count} پرسش واقعی که یک کاربر ایرانی ممکن است "
    "بپرسد بساز. سبک پرسش‌ها: {register}.{adversarial}\n"
    "خروجی را فقط به صورت آرایهٔ JSON بده:\n"
    '[{{"input": "پرسش", "output": "پاسخ درست بر اساس متن", '
    '"contains": ["فقط عبارت‌های خیلی کوتاه و قطعی مثل عدد، مبلغ یا نام '
    'که در هر پاسخ درستی حتماً ظاهر می‌شوند"]}}]'
)
_ADVERSARIAL_FA = (
    " چند پرسش را عمداً سخت کن: ارقام فارسی/لاتین را قاطی کن، تاریخ شمسی به کار "
    "ببر، و یک پرسش را فینگلیش بنویس."
)

_TEXT_SUFFIXES = {".txt", ".md", ".rst", ".html", ".json"}


def _chunks(text: str, size: int = 3000) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def _read_docs(docs) -> list[str]:
    paths = []
    for item in [docs] if isinstance(docs, (str, Path)) else list(docs):
        expanded = glob.glob(str(item)) or [str(item)]
        paths.extend(expanded)
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(
                q for q in path.rglob("*")
                if q.is_file() and q.suffix.lower() in _TEXT_SUFFIXES
            ))
        else:
            files.append(path)
    texts = []
    for path in files:
        if path.suffix.lower() in _TEXT_SUFFIXES | {""}:
            texts.append(path.read_text(encoding="utf-8"))
        else:
            raise ValueError(
                f"{path.suffix} files are not supported yet — convert {path.name} to text/markdown."
            )
    if not texts:
        raise ValueError(f"no documents found at {docs!r}.")
    return texts


class GoldenGenerator:
    """
    GoldenGenerator turns the user's own documentation into Golden
    expectations, ready to feed an AppEvaluator. Generated goldens carry
    their source chunk as `context`, so faithfulness is judged automatically.

    Attributes:
        model (Model | Callable | str, optional): The generator LLM (falls
            back to PARSBENCH_GENERATOR, then PARSBENCH_JUDGE env).
        registers (list[str], optional): Question registers to rotate through
            (default is formal and colloquial Persian).
        adversarial (bool): Mix digit scripts, Jalali dates, and Finglish
            into some questions (default is False).

    Methods:
        generate: Generates goldens from documents.
    """

    def __init__(
        self,
        model: Any = None,
        registers: list[str] | None = None,
        adversarial: bool = False,
    ):
        self.model = model
        self.registers = registers or ["رسمی", "محاوره‌ای"]
        self.adversarial = adversarial

    def generate(self, docs, n: int = 20) -> list[Golden]:
        """
        Generate goldens from the given documents.

        Parameters:
            docs (str | Path | list): A file path, glob pattern, directory, or
                a list of those. Text-like files only (.txt, .md, .rst, .html,
                .json).
            n (int, optional): The number of goldens to generate (default is 20).

        Returns:
            list[Golden]: The generated goldens.
        """
        llm = resolve_model(self.model, "PARSBENCH_GENERATOR", "PARSBENCH_JUDGE")
        if llm is None:
            raise ValueError(
                "generator model not configured — pass model= or set PARSBENCH_JUDGE."
            )
        chunks = [c for text in _read_docs(docs) for c in _chunks(text)]
        per_chunk = max(1, -(-n // len(chunks)))
        goldens: list[Golden] = []
        for i, chunk in enumerate(chunks):
            if len(goldens) >= n:
                break
            prompt = TESTGEN_FA.format(
                chunk=chunk,
                count=min(per_chunk, n - len(goldens)),
                register=self.registers[i % len(self.registers)],
                adversarial=_ADVERSARIAL_FA if self.adversarial else "",
            )
            reply = str(llm(prompt))
            match = re.search(r"\[.*\]", reply, re.DOTALL)
            if not match:
                continue
            try:
                rows = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            for row in rows:
                if not isinstance(row, dict) or not row.get("input"):
                    continue
                goldens.append(
                    Golden(
                        input=row["input"],
                        output=row.get("output"),
                        contains=row.get("contains") or [],
                        context=[chunk],
                        tags=["generated"],
                    )
                )
        return goldens[:n]
