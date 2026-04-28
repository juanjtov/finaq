"""LangGraph state definition for the FINAQ drill-in graph.

Each worker agent fills its corresponding key (fundamentals / filings / news /
risk). `messages` and `errors` use the operator.add reducer so multiple
parallel nodes can append without clobbering each other.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class FinaqState(TypedDict, total=False):
    ticker: str
    thesis: dict[str, Any]
    fundamentals: dict[str, Any]
    filings: dict[str, Any]
    news: dict[str, Any]
    risk: dict[str, Any]
    monte_carlo: dict[str, Any]
    report: str
    messages: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
