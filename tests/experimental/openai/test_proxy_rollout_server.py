"""Unit tests for the proxy rollout server's session key handling."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy.server import SessionData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ADMIN_KEY = "test-admin-key"


@pytest.fixture(autouse=True)
def _reset_server_globals(monkeypatch):
    """Reset all module-level globals before each test."""
    monkeypatch.setattr(srv, "_session_cache", {})
    monkeypatch.setattr(srv, "_api_key_to_session", {})
    monkeypatch.setattr(srv, "_session_to_api_key", {})
    monkeypatch.setattr(srv, "_capacity", 0)
    monkeypatch.setattr(srv, "_admin_api_key", _ADMIN_KEY)
    monkeypatch.setattr(srv, "_lock", threading.Lock())
    monkeypatch.setattr(srv, "_last_cleanup_time", 0.0)


httpx = pytest.importorskip("httpx")

_transport = httpx.ASGITransport(app=srv.app)


def _client():
    return httpx.AsyncClient(transport=_transport, base_url="http://testserver")


def _admin_headers():
    return {"Authorization": f"Bearer {_ADMIN_KEY}"}


# ---------------------------------------------------------------------------
# Tests: start_session with provided api_key
# ---------------------------------------------------------------------------


class TestStartSessionApiKey:
    @pytest.mark.asyncio
    async def test_uses_provided_api_key(self, monkeypatch):
        """Worker returns the caller-provided key instead of generating one."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "my-preferred-key"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_key"] == "my-preferred-key"
        assert srv._api_key_to_session["my-preferred-key"] == data["session_id"]

    @pytest.mark.asyncio
    async def test_generates_key_when_none(self, monkeypatch):
        """No api_key → worker generates a random key (current behaviour)."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
        assert resp.status_code == 200
        key = resp.json()["api_key"]
        assert key != _ADMIN_KEY
        assert len(key) > 10  # random token

    @pytest.mark.asyncio
    async def test_rejects_admin_key_as_session_key(self, monkeypatch):
        """Cannot use the admin key as a session key."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": _ADMIN_KEY},
            )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_cleans_up_finished_session_conflict(self, monkeypatch):
        """Key reuse after a finished session cleans up old mappings."""
        # Pre-seed a finished session with the same key.
        sid_old = "old-session"
        old_session = SessionData(session_id=sid_old)
        old_session.finish()
        srv._session_cache[sid_old] = old_session
        srv._api_key_to_session["reuse-me"] = sid_old
        srv._session_to_api_key[sid_old] = "reuse-me"
        monkeypatch.setattr(srv, "_capacity", 1)

        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "reuse-me"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_key"] == "reuse-me"
        # Old session mapping should be gone; new one present.
        assert srv._api_key_to_session["reuse-me"] == data["session_id"]
        assert data["session_id"] != sid_old

    @pytest.mark.asyncio
    async def test_rejects_active_session_conflict(self, monkeypatch):
        """Key bound to an active (unfinished) session → 409."""
        sid_active = "active-session"
        active_session = SessionData(session_id=sid_active)
        srv._session_cache[sid_active] = active_session
        srv._api_key_to_session["busy-key"] = sid_active
        srv._session_to_api_key[sid_active] = "busy-key"
        monkeypatch.setattr(srv, "_capacity", 1)

        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "busy-key"},
            )
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Tests: end_session returns interaction_count
# ---------------------------------------------------------------------------


class TestEndSessionInteractionCount:
    @pytest.mark.asyncio
    async def test_end_session_returns_interaction_count(self, monkeypatch):
        """end_session response includes interaction_count field."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            # Start a session.
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 200
            api_key = resp.json()["api_key"]

            # End it immediately (0 interactions).
            resp_end = await client.post(
                "/rl/end_session",
                headers={"Authorization": f"Bearer {api_key}"},
                json={},
            )
            assert resp_end.status_code == 200
            data = resp_end.json()
            assert data["interaction_count"] == 0


# ---------------------------------------------------------------------------
# Tests: export_trajectories (requires session_id + admin auth)
# ---------------------------------------------------------------------------


class TestExportTrajectories:
    """Tests for the export_trajectories endpoint.

    The endpoint requires an explicit ``session_id`` in the request body
    and admin-key authentication.  This eliminates routing ambiguity when
    an API key has been reused across sessions.
    """

    @pytest.mark.asyncio
    async def test_export_with_session_id_and_admin_auth(self, monkeypatch):
        """Export succeeds with required session_id + admin key."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 200
            session_id = resp.json()["session_id"]
            api_key = resp.json()["api_key"]

            # End the session so export doesn't block.
            await client.post(
                "/rl/end_session",
                headers={"Authorization": f"Bearer {api_key}"},
                json={},
            )

            resp_export = await client.post(
                "/export_trajectories",
                headers=_admin_headers(),
                json={"session_id": session_id},
            )
            assert resp_export.status_code == 200
            assert "interactions" in resp_export.json()

    @pytest.mark.asyncio
    async def test_export_rejects_non_admin_key(self, monkeypatch):
        """Export requires admin auth; a session key must be rejected."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 200
            session_id = resp.json()["session_id"]
            api_key = resp.json()["api_key"]

            await client.post(
                "/rl/end_session",
                headers={"Authorization": f"Bearer {api_key}"},
                json={},
            )

            resp_export = await client.post(
                "/export_trajectories",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"session_id": session_id},
            )
            assert resp_export.status_code == 403

    @pytest.mark.asyncio
    async def test_export_rejects_missing_session_id(self, monkeypatch):
        """Omitting session_id from body triggers a validation error (422)."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp_export = await client.post(
                "/export_trajectories",
                headers=_admin_headers(),
                json={},
            )
            assert resp_export.status_code == 422

    @pytest.mark.asyncio
    async def test_export_survives_key_remap(self, monkeypatch):
        """Explicit session_id resolves correctly even after key remapping.

        After a session refresh the API key maps to the NEW session.
        Because export uses the explicit session_id, it still targets
        the OLD (completed) session without blocking.
        """
        monkeypatch.setattr(srv, "_capacity", 2)
        async with _client() as client:
            # Start first session with a specific key.
            resp1 = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "task-0", "api_key": "shared-key"},
            )
            assert resp1.status_code == 200
            session_id_old = resp1.json()["session_id"]

            # End the first session.
            await client.post(
                "/rl/end_session",
                headers={"Authorization": "Bearer shared-key"},
                json={},
            )

            # Start a second session reusing the same key.
            resp2 = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "task-1", "api_key": "shared-key"},
            )
            assert resp2.status_code == 200
            session_id_new = resp2.json()["session_id"]
            assert session_id_new != session_id_old

            # API key now points to the NEW session.
            assert srv._api_key_to_session["shared-key"] == session_id_new

            # Export the OLD session by session_id — unaffected by the remap.
            resp_export = await client.post(
                "/export_trajectories",
                headers=_admin_headers(),
                json={"session_id": session_id_old},
            )
            assert resp_export.status_code == 200
            assert "interactions" in resp_export.json()


class _ContextLimitCreate:
    async def create(self, *, messages, areal_cache=None, temperature=1.0, top_p=1.0):
        from areal.experimental.openai.types import ContextLengthExceededError

        raise ContextLengthExceededError(
            prompt_len=8, limit_name="max_total_tokens", limit=8
        )


class _PlainValueErrorCreate:
    async def create(self, *, messages, areal_cache=None, temperature=1.0, top_p=1.0):
        raise ValueError("areal_context_limit appears in an unrelated ValueError")


class _CapturingChatCreate:
    def __init__(self):
        self.calls = []

    async def create(
        self,
        *,
        messages,
        areal_cache=None,
        temperature=1.0,
        top_p=1.0,
        max_completion_tokens=None,
        max_total_tokens=None,
        extra_body=None,
        stream=False,
    ):
        self.calls.append(
            {
                "temperature": temperature,
                "top_p": top_p,
                "max_completion_tokens": max_completion_tokens,
                "max_total_tokens": max_total_tokens,
                "extra_body": extra_body,
                "stream": stream,
                "areal_cache": areal_cache,
            }
        )
        if stream:
            return _empty_chunk_stream()
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [],
        }


def _chat_payload(**kwargs):
    payload = {"model": "ignored", "messages": [{"role": "user", "content": "x"}]}
    payload.update(kwargs)
    return payload


async def _empty_chunk_stream():
    if False:
        yield None


class _FakeOpenAIClient:
    def __init__(self, create):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create.create))


def _install_session(monkeypatch, create):
    session_id = "session-0"
    api_key = "session-key"
    srv._session_cache[session_id] = SessionData(session_id=session_id)
    srv._api_key_to_session[api_key] = session_id
    srv._session_to_api_key[session_id] = api_key
    monkeypatch.setattr(srv, "_openai_client", _FakeOpenAIClient(create))
    return {"Authorization": f"Bearer {api_key}"}


@pytest.mark.asyncio
async def test_chat_completions_context_limit_returns_structured_http_400(monkeypatch):
    """Typed context preflight errors are user-input 400s, not infra 500s."""

    headers = _install_session(monkeypatch, _ContextLimitCreate())

    async with _client() as client:
        resp = await client.post(
            "/chat/completions",
            headers=headers,
            json=_chat_payload(),
        )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["code"] == "context_length_exceeded"
    assert "areal_context_limit" in detail["message"]


@pytest.mark.asyncio
async def test_chat_completions_plain_value_error_with_marker_stays_500(monkeypatch):
    """Only the typed exception is classified as context length exceeded."""

    headers = _install_session(monkeypatch, _PlainValueErrorCreate())

    async with _client() as client:
        resp = await client.post(
            "/chat/completions",
            headers=headers,
            json=_chat_payload(),
        )

    assert resp.status_code == 500
    assert (
        resp.json()["detail"]
        == "areal_context_limit appears in an unrelated ValueError"
    )


@pytest.mark.asyncio
async def test_chat_completions_standard_request_still_reaches_client(monkeypatch):
    """OpenAI-standard chat requests still pass through unchanged."""

    create = _CapturingChatCreate()
    headers = _install_session(monkeypatch, create)

    async with _client() as client:
        resp = await client.post(
            "/chat/completions",
            headers=headers,
            json=_chat_payload(
                temperature=0.2,
                top_p=0.9,
                max_completion_tokens=7,
            ),
        )

    assert resp.status_code == 200
    assert len(create.calls) == 1
    assert create.calls[0]["temperature"] == 0.2
    assert create.calls[0]["top_p"] == 0.9
    assert create.calls[0]["max_completion_tokens"] == 7
    assert create.calls[0]["extra_body"] is None
    assert create.calls[0]["max_total_tokens"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expect_stream", "expect_max_total_tokens"),
    [
        (
            _chat_payload(
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "max_total_tokens": 32767,
                },
            ),
            False,
            32767,
        ),
        (
            _chat_payload(
                stream=True,
                chat_template_kwargs={"enable_thinking": False},
                max_total_tokens=321,
            ),
            True,
            321,
        ),
    ],
)
async def test_chat_completions_preserves_areal_extensions(
    monkeypatch, payload, expect_stream, expect_max_total_tokens
):
    """Nested and SDK-flattened extensions reach the AReaL chat client."""

    create = _CapturingChatCreate()
    headers = _install_session(monkeypatch, create)

    async with _client() as client:
        resp = await client.post(
            "/chat/completions",
            headers=headers,
            json=payload,
        )
        body = resp.text

    assert resp.status_code == 200
    if expect_stream:
        assert "data: [DONE]" in body
    assert len(create.calls) == 1
    assert create.calls[0]["stream"] is expect_stream
    assert create.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert create.calls[0]["max_total_tokens"] == expect_max_total_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        (
            _chat_payload(chat_template_kwargs="bad"),
            "chat_template_kwargs must be an object",
        ),
        (
            _chat_payload(extra_body="bad"),
            "extra_body must be an object when provided",
        ),
        (
            _chat_payload(
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                chat_template_kwargs={"enable_thinking": False},
            ),
            "conflicting chat_template_kwargs values in top-level body and extra_body",
        ),
    ],
)
async def test_chat_completions_rejects_bad_areal_extensions(
    monkeypatch, payload, detail
):
    """Malformed or conflicting known extensions fail loudly."""

    create = _CapturingChatCreate()
    headers = _install_session(monkeypatch, create)

    async with _client() as client:
        resp = await client.post(
            "/chat/completions",
            headers=headers,
            json=payload,
        )

    assert resp.status_code == 400
    assert resp.json()["detail"] == detail
    assert create.calls == []
