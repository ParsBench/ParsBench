"""Framework and platform integrations. All adapters are duck-typed — no
framework dependency is required to import them; each converts that
framework's run/trace object into a parsbench Trace."""

from . import adk, agno, langfuse, langgraph, openai_agents, otel, pydantic_ai

__all__ = ["adk", "agno", "langfuse", "langgraph", "openai_agents", "otel", "pydantic_ai"]
