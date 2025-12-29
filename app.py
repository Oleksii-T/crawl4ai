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

    try:
        result = await run_crawl(
            url=payload.url,
            instructions=payload.instructions,
            schema_input=payload.schema,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return CrawlResponse(**result)
