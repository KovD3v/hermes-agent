"""Native Codex search must remain a tool after SQLite persistence and reload."""
import json

import pytest

from agent.transports.codex_event_projector import CodexEventProjector


@pytest.mark.parametrize("search", [
    {"query": "Hermes docs", "action": {"type": "search"}},
    {"action": {"type": "search", "query": "Hermes docs"}},
    {"query": "Hermes docs", "action": {"type": "openPage", "url": "https://example.org"}},
    {"action": {"type": "findInPage", "url": "https://example.org", "pattern": "Codex"}},
    {"action": None},
])
def test_web_search_remains_a_tool_after_persistence_and_reload(tmp_path, search):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from agent.codex_runtime import (
        _codex_item_to_tool_name, make_codex_app_server_event_bridge,
    )
    from run_agent import AIAgent
    from hermes_state import SessionDB
    from tui_gateway.server import _history_to_messages

    item = {"type": "webSearch", "id": "native-search", "status": "completed", **search}
    note = {"method": "item/completed", "params": {"item": item}}
    projection = CodexEventProjector().project(note)
    assert CodexEventProjector().project(note).messages == projection.messages
    live = SimpleNamespace(tool_start_callback=Mock())
    make_codex_app_server_event_bridge(live)({"method": "item/started", "params": {"item": item}})
    store = AIAgent.__new__(AIAgent)
    store._session_db = SessionDB(tmp_path / "session.db")
    store.session_id = "web-search-replay"
    store._session_db.create_session(store.session_id, source="gui")
    store._session_db_created = True
    store._last_flushed_db_idx = 0
    try:
        store._flush_messages_to_session_db(
            [{"role": "user", "content": "Search docs."}] + projection.messages
            + [{"role": "assistant", "content": "Done."}]
        )
        raw = store._session_db.get_messages_as_conversation(store.session_id)
        display = raw
    finally:
        store._session_db.close()
    messages = _history_to_messages(display)
    assert projection.is_tool_iteration
    assistant, tool = projection.messages
    assert assistant["content"] is None
    call = assistant["tool_calls"][0]
    assert call["function"]["name"] == "codex_web_search"
    assert tool["tool_call_id"] == call["id"] == live.tool_start_callback.call_args.args[0]
    args = json.loads(call["function"]["arguments"])
    assert args["query"] == (search.get("query") or (search.get("action") or {}).get("query") or "")
    result = json.loads(tool["content"])
    assert result["provider"] == "codex"
    assert result["status"] == item["status"]

    assert [m["role"] for m in messages] == ["user", "tool", "assistant"]
    assert messages[1]["name"] == call["function"]["name"] == _codex_item_to_tool_name(item)
    reloaded_call = next(m for m in raw if m.get("tool_calls"))["tool_calls"][0]
    assert json.loads(reloaded_call["function"]["arguments"]) == args
    assert reloaded_call["id"] == call["id"]
    assert next(m for m in raw if m["role"] == "tool")["content"] == tool["content"]
