import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from crawler_service import run_crawl

app = FastAPI(title="crawl4ai api")
BEARER_TOKEN = os.getenv("CRAWL4AI_BEARER_TOKEN")


class CrawlRequest(BaseModel):
    url: str
    instructions: str
    schema: Dict[str, Any] | str
    proxy_url: Optional[str] = None
    proxy_port: Optional[int] = None
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None


class CrawlResponse(BaseModel):
    success: bool
    data: Optional[Any]
    error: Optional[str]


@app.post("/crawl4ai", response_model=CrawlResponse)
async def crawl4ai_endpoint(payload: CrawlRequest, request: Request) -> CrawlResponse:
    if BEARER_TOKEN:
        auth_header = request.headers.get("authorization", "")
        if auth_header != f"Bearer {BEARER_TOKEN}":
            raise HTTPException(status_code=401, detail="Invalid bearer token.")

    proxy_fields = [
        payload.proxy_url,
        payload.proxy_port,
        payload.proxy_username,
        payload.proxy_password,
    ]
    proxy_provided = any(field is not None for field in proxy_fields)
    if proxy_provided and not all(field is not None for field in proxy_fields):
        raise HTTPException(
            status_code=400,
            detail=(
                "proxy_url, proxy_port, proxy_username, and proxy_password must all be "
                "provided together."
            ),
        )

    proxy_config = None
    if proxy_provided:
        proxy_config = {
            "server": f"{payload.proxy_url.rstrip('/') if payload.proxy_url else ''}:{payload.proxy_port}",
            "username": payload.proxy_username,
            "password": payload.proxy_password,
        }

    try:
        result = await run_crawl(
            url=payload.url,
            instructions=payload.instructions,
            schema_input=payload.schema,
            proxy_config=proxy_config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return CrawlResponse(**result)
