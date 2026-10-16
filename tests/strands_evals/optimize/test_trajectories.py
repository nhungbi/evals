from datetime import datetime, timedelta, timezone

from strands_evals.optimize.trajectories import render_trajectory
from strands_evals.types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    InferenceSpan,
    Session,
    SpanInfo,
    TextContent,
    ToolCall,
    ToolExecutionSpan,
    ToolResult,
    Trace,
    UserMessage,
)

T0 = datetime(2026, 10, 15, tzinfo=timezone.utc)


def _info(seconds: int) -> SpanInfo:
    start = T0 + timedelta(seconds=seconds)
    return SpanInfo(session_id="s", start_time=start, end_time=start + timedelta(milliseconds=500))


def _turn(trace_id: str, start: int, prompt: str, response: str, tools=()) -> Trace:
    spans = [
        AgentInvocationSpan(span_info=_info(start), user_prompt=prompt, agent_response=response, available_tools=[])
    ]
    for offset, (name, args, content, error) in enumerate(tools, start=1):
        spans.append(
            ToolExecutionSpan(
                span_info=_info(start + offset),
                tool_call=ToolCall(name=name, arguments=args),
                tool_result=ToolResult(content=content, error=error),
            )
        )
    return Trace(spans=spans, trace_id=trace_id, session_id="s")


def _session() -> Session:
    return Session(
        session_id="s",
        traces=[
            _turn("t1", 0, "My ticket ID is TK-1234.", "Noted."),
            _turn("t2", 10, "Look up my order.", "It shipped.", tools=[("lookup", {"id": 7}, "shipped", None)]),
            _turn("t3", 20, "What was my ticket ID?", "I don't know.", tools=[("search", {"q": "TK"}, "", "timeout")]),
        ],
    )


def test_session_renders_one_block_per_turn_with_tools_in_order():
    text = render_trajectory(_session())

    assert text.splitlines() == [
        "--- turn 1 ---",
        "[user] My ticket ID is TK-1234.",
        "[assistant] Noted.",
        "--- turn 2 ---",
        "[user] Look up my order.",
        '[tool lookup] {"id": 7}',
        "[tool result] shipped",
        "[assistant] It shipped.",
        "--- turn 3 ---",
        "[user] What was my ticket ID?",
        '[tool search] {"q": "TK"}',
        "[tool result] ERROR timeout:",
        "[assistant] I don't know.",
    ]


def test_session_dump_from_a_stored_report_renders_the_same():
    session = _session()
    assert render_trajectory(session.model_dump(mode="json")) == render_trajectory(session)


def test_trace_without_agent_span_falls_back_to_the_last_model_call():
    inference = InferenceSpan(
        span_info=_info(0),
        messages=[UserMessage(content=[TextContent(text="hi")]), AssistantMessage(content=[TextContent(text="hello")])],
    )
    session = Session(session_id="s", traces=[Trace(spans=[inference], trace_id="t", session_id="s")])

    assert render_trajectory(session).splitlines() == ["--- turn 1 ---", "[user] hi", "[assistant] hello"]


def test_strands_message_list_renders_text_tool_use_and_tool_result():
    messages = [
        {"role": "user", "content": [{"text": "find it"}]},
        {"role": "assistant", "content": [{"toolUse": {"name": "search", "toolUseId": "1", "input": {"q": "x"}}}]},
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "1", "status": "success", "content": [{"text": "found"}]}}],
        },
        {"role": "assistant", "content": [{"text": "here"}]},
    ]

    assert render_trajectory(messages).splitlines() == [
        "[user] find it",
        '[assistant -> tool search] {"q": "x"}',
        "[tool result success] found",
        "[assistant] here",
    ]


def test_none_renders_empty_and_other_values_render_as_text():
    assert render_trajectory(None) == ""
    assert render_trajectory(["search", "fetch"]) == "search\nfetch"
    assert render_trajectory({"steps": 2}) == '{"steps": 2}'


def test_long_trajectory_keeps_head_and_tail_within_the_cap():
    messages = [{"role": "user", "content": [{"text": f"turn {i} " + "x" * 50}]} for i in range(200)]

    text = render_trajectory(messages, max_chars=1_000)

    assert len(text) <= 1_000
    assert text.startswith("[user] turn 0 ")
    assert text.rstrip().endswith("x" * 10)
    assert "turn 199" in text
    assert "chars omitted" in text


def test_each_line_is_clipped():
    text = render_trajectory([{"role": "user", "content": [{"text": "y" * 5_000}]}], max_chars=10_000)

    assert len(text) < 700
    assert text.endswith(" ...")
