"""Judge resolution: set once, use everywhere.

Accepted forms, in resolution order:
- explicit callable `prompt -> str` or object with get_prompt_completion
- explicit model-name string -> OpenAI-compatible client from env
- PARSBENCH_JUDGE env var (model name), with PARSBENCH_JUDGE_BASE_URL /
  PARSBENCH_JUDGE_API_KEY (falling back to OPENAI_BASE_URL / OPENAI_API_KEY)
"""

import os


def resolve_model(spec, *env_vars: str, temperature: float | None = 0):
    """Resolve a model spec into a `prompt -> str` callable.

    spec: callable / object with get_prompt_completion / model-name string /
    None (falls back to the given env vars in order).
    temperature: sampling temperature for string specs; None leaves the
    provider default (used for the user simulator, which needs variety).
    """
    if spec is None:
        spec = next((os.environ[e] for e in env_vars if os.environ.get(e)), None)
        if spec is None:
            return None
    if not isinstance(spec, str):
        # unwrap parsbench Model-style objects here, once, so every consumer
        # gets a plain `prompt -> str` callable
        return getattr(spec, "get_prompt_completion", spec)

    import openai

    # ponytail: one gateway for judge & simulator alike; per-role base URLs
    # can come later if anyone actually runs them on different endpoints.
    # Batch evals need to ride out 429s/timeouts without hanging CI forever;
    # override with PARSBENCH_MAX_RETRIES / PARSBENCH_TIMEOUT (seconds).
    client = openai.OpenAI(
        base_url=os.environ.get("PARSBENCH_JUDGE_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("PARSBENCH_JUDGE_API_KEY")
        or os.environ.get("OPENAI_API_KEY", "-"),
        max_retries=int(os.environ.get("PARSBENCH_MAX_RETRIES", 5)),
        timeout=float(os.environ.get("PARSBENCH_TIMEOUT", 120)),
    )
    model = spec
    sampling: dict = {} if temperature is None else {"temperature": temperature}

    def _rejects_temperature(exc: openai.BadRequestError) -> bool:
        # prefer the SDK's structured error field; fall back to message text
        # for gateways/proxies that word the 400 differently
        return getattr(exc, "param", None) == "temperature" or "temperature" in str(exc)

    def _call(prompt: str) -> str:
        messages: list = [{"role": "user", "content": prompt}]
        sent = dict(sampling)  # snapshot: `sampling` is shared across threads
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, **sent
            )
        except openai.BadRequestError as exc:
            # reasoning models (gpt-5 family, o-series) reject explicit temperature
            if "temperature" not in sent or not _rejects_temperature(exc):
                raise
            sampling.pop("temperature", None)  # later calls skip it; racing pops are fine
            response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content or ""

    return _call


def resolve_judge(judge):
    return resolve_model(judge, "PARSBENCH_JUDGE")
