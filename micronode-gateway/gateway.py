import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI()
OLLAMA_URL = "http://host.docker.internal:11434"
LITELLM_URL = "http://host.docker.internal:4000"
API_KEY = "micronode-secret-key"
LITELLM_API_KEY = "sk-micronode-router-key"


def _build_upstream_headers(request: Request, is_litellm: bool) -> list[tuple[bytes, bytes]]:
    if not is_litellm:
        return list(request.headers.raw)

    headers = [
        (k, v)
        for k, v in request.headers.raw
        if k.lower() not in (b"authorization", b"host")
    ]
    headers.append((b"authorization", f"Bearer {LITELLM_API_KEY}".encode("utf-8")))
    return headers


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_request(path: str, request: Request):
    if request.headers.get("authorization", "") != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized Node Access")

    is_litellm = request.url.path == "/v1" or request.url.path.startswith("/v1/")
    base_url = LITELLM_URL if is_litellm else OLLAMA_URL
    upstream_headers = _build_upstream_headers(request, is_litellm)

    client = httpx.AsyncClient(base_url=base_url)
    url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
    req = client.build_request(
        request.method,
        url,
        headers=upstream_headers,
        content=await request.body(),
    )

    resp = await client.send(req, stream=True)

    async def stream_response():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        stream_response(),
        status_code=resp.status_code,
        headers=resp.headers,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
