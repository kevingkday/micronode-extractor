import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI()
OLLAMA_URL = "http://host.docker.internal:11434"
API_KEY = "micronode-secret-key"

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT"])
async def proxy_ollama(path: str, request: Request):
    if request.headers.get("authorization", "") != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized Node Access")
    
    client = httpx.AsyncClient(base_url=OLLAMA_URL)
    url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
    req = client.build_request(request.method, url, headers=request.headers.raw, content=await request.body())
    
    resp = await client.send(req, stream=True)
    return StreamingResponse(resp.aiter_raw(), status_code=resp.status_code, headers=resp.headers)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

