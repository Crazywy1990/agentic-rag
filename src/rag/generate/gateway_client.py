import httpx
import structlog

from rag.config import settings

DEFAULT_MODEL = "claude-haiku-4-5"  # gateway-routed; W2: A/B stronger generator
TIMEOUT = httpx.Timeout(60.0, connect=10.0)

log = structlog.get_logger()


class GatewayClient:
    """Client for the Project-1 LLM gateway. All generation flows through it:
    this service inherits its semantic caching + observability for free."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self._base_url = (base_url or settings.gateway_base_url).rstrip("/")
        self._api_key = api_key or settings.gateway_api_key

    def complete(self, system: str, user: str, model: str = DEFAULT_MODEL) -> str:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        resp = httpx.post(
            f"{self._base_url}/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,  # eval reproducibility + cache hits
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        log.info(
            "gateway_completion",
            model=model,
            cache=resp.headers.get("x-cache", "unknown"),
        )
        return resp.json()["choices"][0]["message"]["content"]
