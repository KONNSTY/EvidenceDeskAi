import pytest

from evidencedesk.config import Settings
from evidencedesk.llm import OpenAIClient


def test_client_is_not_created_on_construction() -> None:
    client = OpenAIClient(Settings(openai_api_key=None))

    with pytest.raises(RuntimeError, match="ED_OPENAI_API_KEY"):
        _ = client.client


def test_client_is_created_from_configured_key() -> None:
    client = OpenAIClient(Settings(openai_api_key="sk-test"))

    assert client.client is client.client
