"""Render a rollout's trajectory as compact text for the proposer.

A task can return `{"output": ..., "trajectory": ...}`, and `Experiment` stores the trajectory in
the report as `actual_trajectory`. The optimizer renders it here so a reflection record shows
what the agent did, not only what it finally answered. Three shapes are understood:

- a `Session` (or its JSON dump), as `TracedHandler` / `@eval_task` produce: one block per trace,
  with the user prompt, each tool call and its result, and the agent's response;
- a list of messages in the strands format (`{"role", "content": [blocks]}`), such as
  `agent.messages`;
- anything else, rendered as text.

Every line is clipped on its own, then a transcript that is still too long keeps its head and
tail: the early turns hold the context a failure often depends on, and the last ones hold the
answer.
"""

import json

from pydantic import BaseModel
from typing_extensions import Any

from ..types.trace import SpanType

DEFAULT_MAX_TRAJECTORY_CHARS = 5_000
_MAX_LINE_CHARS = 600


def render_trajectory(trajectory: Any, max_chars: int = DEFAULT_MAX_TRAJECTORY_CHARS) -> str:
    """Render `trajectory` as at most `max_chars` characters, or `""` when there is none."""
    if trajectory is None:
        return ""
    if isinstance(trajectory, BaseModel):
        trajectory = trajectory.model_dump(mode="json")
    if isinstance(trajectory, dict) and isinstance(trajectory.get("traces"), list):
        lines = _session_lines(trajectory)
    elif isinstance(trajectory, list):
        lines = [line for item in trajectory for line in _message_lines(item)]
    else:
        lines = [_clip(_as_text(trajectory))]
    return _head_and_tail("\n".join(lines), max_chars)


def _session_lines(session: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for number, trace in enumerate(session["traces"], start=1):
        spans = sorted(trace.get("spans") or [], key=lambda s: (s.get("span_info") or {}).get("start_time") or "")
        agents = [s for s in spans if s.get("span_type") == SpanType.AGENT_INVOCATION.value]
        tools = [s for s in spans if s.get("span_type") == SpanType.TOOL_EXECUTION.value]
        lines.append(f"--- turn {number} ---")
        if agents:
            lines.append(f"[user] {_clip(agents[0].get('user_prompt', ''))}")
        else:
            # No agent span (a custom mapper): fall back to the last model call's messages.
            inference = [s for s in spans if s.get("span_type") == SpanType.INFERENCE.value]
            if inference:
                lines.extend(line for m in inference[-1].get("messages") or [] for line in _message_lines(m))
        for span in tools:
            call, result = span.get("tool_call") or {}, span.get("tool_result") or {}
            lines.append(f"[tool {call.get('name', '?')}] {_clip(_as_text(call.get('arguments')))}")
            outcome = f"ERROR {result['error']}: " if result.get("error") else ""
            lines.append(f"[tool result] {_clip(outcome + _as_text(result.get('content', '')))}")
        if agents:
            lines.append(f"[assistant] {_clip(agents[-1].get('agent_response', ''))}")
    return lines


def _message_lines(message: Any) -> list[str]:
    """One line per content block of a strands-format message, or the item as text."""
    if not isinstance(message, dict) or "content" not in message:
        return [_clip(_as_text(message))]
    role = message.get("role", "?")
    content = message["content"]
    if isinstance(content, str):
        return [f"[{role}] {_clip(content)}"]
    lines = []
    for block in content or []:
        if not isinstance(block, dict):
            lines.append(f"[{role}] {_clip(_as_text(block))}")
        elif "text" in block:
            lines.append(f"[{role}] {_clip(block['text'])}")
        elif "toolUse" in block:
            use = block["toolUse"]
            lines.append(f"[{role} -> tool {use.get('name', '?')}] {_clip(_as_text(use.get('input')))}")
        elif "toolResult" in block:
            result = block["toolResult"]
            parts = [item.get("text", _as_text(item)) for item in result.get("content") or [] if isinstance(item, dict)]
            lines.append(f"[tool result {result.get('status', '')}] {_clip(' '.join(parts))}")
        elif "name" in block and "arguments" in block:  # the trace types' ToolCallContent
            lines.append(f"[{role} -> tool {block['name']}] {_clip(_as_text(block['arguments']))}")
        elif "content" in block and block.get("content_type") == "tool_result":  # ToolResultContent
            lines.append(f"[tool result] {_clip(_as_text(block['content']))}")
        else:
            lines.append(f"[{role}] <{next(iter(block), 'empty')}>")
    return lines


def _head_and_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n... [{} chars omitted] ...\n"
    keep = max(0, max_chars - len(marker.format(len(text))))
    head, tail = keep - keep // 2, keep // 2
    return text[:head] + marker.format(len(text) - head - tail) + (text[len(text) - tail :] if tail else "")


def _clip(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= _MAX_LINE_CHARS else text[:_MAX_LINE_CHARS] + " ..."


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=repr)
    except (TypeError, ValueError):
        return str(value)
