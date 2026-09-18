"""Arena deployments may serve LLM traffic separately from the control API."""

import httpx
import pytest

from examples.swe.arena_client import ArenaOpenAPIClient


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "override,expected",
    [
        (None, "https://control.example/arena/api"),
        ("https://llm.example/api/", "https://llm.example/api"),
    ],
)
@pytest.mark.asyncio
async def test_registration_uses_gateway_url_without_changing_control_plane(
    monkeypatch, asynchronous, override, expected
):
    if override is None:
        monkeypatch.delenv("ARENA_LLM_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("ARENA_LLM_BASE_URL", override)
    client = ArenaOpenAPIClient(
        base_url="https://control.example/arena/", api_token="test-token"
    )
    model = "stream-areal-test"

    def handler(request):
        assert str(request.url) == "https://control.example/arena/openapi/v1/llm/models"
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(200, json={"model_name": model})

    kwargs = dict(
        model_name=model,
        upstream_base_url="http://proxy.example",
        upstream_api_key="session-key",
        deployment_id="deployment",
    )
    transport = httpx.MockTransport(handler)
    if asynchronous:
        async with httpx.AsyncClient(transport=transport) as http_client:
            result = await client.register_llm_proxy_async(**kwargs, client=http_client)
    else:
        with httpx.Client(transport=transport) as http_client:
            result = client.register_llm_proxy(**kwargs, client=http_client)
    assert result == (expected, model)


@pytest.mark.parametrize(
    "value",
    [
        "/api",
        "ftp://llm.example/api",
        "https://",
        "https://user:secret@llm.example",
        "https://llm.example/api?key=secret",
        "https://llm.example/api#fragment",
        "https://llm.example:bad/api",
        "https://llm.example:70000/api",
        "https://llm. example/api",
    ],
)
def test_invalid_gateway_url_is_rejected_without_echoing_credentials(
    monkeypatch, value
):
    monkeypatch.setenv("ARENA_LLM_BASE_URL", value)
    with pytest.raises(ValueError, match="Arena LLM base URL") as error:
        ArenaOpenAPIClient(base_url="https://control.example", api_token="test-token")
    assert "secret" not in str(error.value)
