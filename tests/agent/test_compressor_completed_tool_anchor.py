"""Regression for completed newest tool-plan anchoring during compaction.

The last-assistant anchor exists to keep a user-visible final reply readable and
must also preserve genuinely pending tool work. A tool-planning assistant whose
adjacent tool round has already completed is different: it is historical work
and should be eligible for the summary instead of overriding newer summary
state merely because it is the newest assistant row.

Provider tool-call ids are not execution identity. The repeated-id case below
uses Hermes' persisted per-occurrence tool-call UIDs so an older completed
occurrence cannot make a newer unanswered occurrence look complete.
"""

from __future__ import annotations

from agent.context_compressor import ContextCompressor
from agent.message_metadata import TOOL_CALL_UID, TOOL_CALL_UIDS


def _compressor() -> ContextCompressor:
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor.quiet_mode = True
    return compressor


def _tool_call(call_id: str, name: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def test_last_assistant_anchor_distinguishes_completed_pending_and_visible_reply():
    compressor = _compressor()

    completed = [
        {"role": "user", "content": "audit the project"},
        {
            "role": "assistant",
            "content": "I still need to inspect App.cs",
            "tool_calls": [_tool_call("call-plan", "read_file")],
            TOOL_CALL_UIDS: {"call-plan": "occ-completed"},
        },
        {
            "role": "tool",
            "content": "App.cs contents",
            "tool_call_id": "call-plan",
            TOOL_CALL_UID: "occ-completed",
        },
        {"role": "user", "content": "continue with the analysis"},
    ]
    completed_cut = len(completed)
    assert compressor._ensure_last_assistant_message_in_tail(
        completed, cut_idx=completed_cut, head_end=0
    ) == completed_cut, (
        "a fully completed newest tool plan should not be pulled back into the "
        "protected tail merely because it is the newest assistant row"
    )

    pending = [
        {"role": "user", "content": "audit the project"},
        {
            "role": "assistant",
            "content": "Inspect both files before analysis",
            "tool_calls": [
                _tool_call("call-a", "read_file"),
                _tool_call("call-b", "read_file"),
            ],
            TOOL_CALL_UIDS: {
                "call-a": "occ-a",
                "call-b": "occ-b",
            },
        },
        {
            "role": "tool",
            "content": "first file",
            "tool_call_id": "call-a",
            TOOL_CALL_UID: "occ-a",
        },
    ]
    pending_cut = len(pending)
    assert compressor._ensure_last_assistant_message_in_tail(
        pending, cut_idx=pending_cut, head_end=0
    ) <= 1, "an unanswered tool round must remain protected atomically"

    visible_reply = [
        {"role": "user", "content": "summarize the result"},
        {"role": "assistant", "content": "The audit is complete."},
        {"role": "user", "content": "next question"},
    ]
    assert compressor._ensure_last_assistant_message_in_tail(
        visible_reply, cut_idx=len(visible_reply), head_end=0
    ) <= 1, "a normal visible final assistant reply must remain protected"


def test_reused_provider_id_does_not_complete_newer_unanswered_occurrence():
    compressor = _compressor()

    messages = [
        {"role": "user", "content": "inspect the first file"},
        {
            "role": "assistant",
            "content": "Reading the first file",
            "tool_calls": [_tool_call("call-reused", "read_file")],
            TOOL_CALL_UIDS: {"call-reused": "occ-old"},
        },
        {
            "role": "tool",
            "content": "old result",
            "tool_call_id": "call-reused",
            TOOL_CALL_UID: "occ-old",
        },
        {"role": "user", "content": "now inspect the second file"},
        {
            "role": "assistant",
            "content": "Reading the second file",
            "tool_calls": [_tool_call("call-reused", "read_file")],
            TOOL_CALL_UIDS: {"call-reused": "occ-new"},
        },
    ]

    cut = compressor._ensure_last_assistant_message_in_tail(
        messages, cut_idx=len(messages), head_end=0
    )
    assert cut <= 4, (
        "the older result reusing the same provider tool-call id must not make "
        "the newer per-occurrence tool call look complete"
    )
