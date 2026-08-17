"""Push an AppEvaluationResult's scores to Langfuse (self-hosted or cloud)
via the public ingestion API — no langfuse SDK required, just requests + env
keys: LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY."""

import os
import uuid
from datetime import datetime, timezone


def push(evaluation_result, *, name: str = "parsbench", host: str | None = None,
         public_key: str | None = None, secret_key: str | None = None,
         _post=None):
    host = (host or os.environ["LANGFUSE_HOST"]).rstrip("/")
    public_key = public_key or os.environ["LANGFUSE_PUBLIC_KEY"]
    secret_key = secret_key or os.environ["LANGFUSE_SECRET_KEY"]

    now = datetime.now(timezone.utc).isoformat()
    trace_id = str(uuid.uuid4())
    batch = [
        {
            "id": str(uuid.uuid4()),
            "type": "trace-create",
            "timestamp": now,
            "body": {"id": trace_id, "name": name},
        }
    ]
    for golden_result in evaluation_result.golden_results:
        for check_result in golden_result.check_results:
            if check_result.skipped:
                continue
            batch.append(
                {
                    "id": str(uuid.uuid4()),
                    "type": "score-create",
                    "timestamp": now,
                    "body": {
                        "id": str(uuid.uuid4()),
                        "traceId": trace_id,
                        "name": check_result.check,
                        "value": check_result.score,
                        "comment": f"{golden_result.golden_name}"
                        + (f" — {check_result.reason}" if check_result.reason else ""),
                    },
                }
            )

    if _post is None:  # pragma: no cover - exercised via injection in tests
        import requests

        _post = requests.post
    response = _post(
        f"{host}/api/public/ingestion",
        json={"batch": batch},
        auth=(public_key, secret_key),
        timeout=30,
    )
    response.raise_for_status()
    return trace_id
