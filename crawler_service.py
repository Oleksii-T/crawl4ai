import json
import os
from typing import Any, Dict, Optional

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMConfig,
    ProxyConfig,
)
from dotenv import load_dotenv

from debug_extraction import DebugLLMExtractionStrategy


def _schema_from_mapping(mapping: Dict[str, str]) -> Dict[str, Any]:
    return {
        "title": "CustomSchema",
        "type": "object",
        "properties": {key: {"type": value} for key, value in mapping.items()},
        "required": list(mapping.keys()),
    }


def resolve_schema(schema_input: Any) -> Dict[str, Any]:
    if not schema_input:
        raise ValueError("Missing schema payload.")

    if isinstance(schema_input, str):
        try:
            parsed = json.loads(schema_input)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid schema JSON: {exc}")
    elif isinstance(schema_input, dict):
        parsed = schema_input
    else:
        raise ValueError("Schema must be a JSON object or a JSON string.")

    if isinstance(parsed, dict):
        if "type" in parsed or "properties" in parsed:
            return parsed
        if parsed and all(isinstance(value, str) for value in parsed.values()):
            return _schema_from_mapping(parsed)

    raise ValueError(
        "Invalid schema: expected a JSON schema object or shorthand mapping like "
        '{"title":"string","url":"string"}.'
    )


async def run_crawl(
    url: str,
    instructions: str,
    schema_input: Any,
    proxy_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    load_dotenv()
    os.environ.setdefault("CRAWL4_AI_BASE_DIRECTORY", os.getcwd())
    os.makedirs(os.path.join(os.getcwd(), ".crawl4ai"), exist_ok=True)

    proxy_cfg_obj = None
    if proxy_config:
        proxy_cfg_obj = ProxyConfig(
            server=proxy_config.get("server"),
            username=proxy_config.get("username"),
            password=proxy_config.get("password"),
        )

    llm_strategy = DebugLLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="deepseek/deepseek-chat",
            api_token=os.getenv("DEEPSEEK_API"),
            temperature=0.0,
            max_tokens=800,
        ),
        schema=resolve_schema(schema_input),
        extraction_type="schema",
        instruction=instructions,
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",
    )

    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        process_iframes=False,
        remove_overlay_elements=True,
        exclude_external_links=True,
        proxy_config=proxy_cfg_obj,
    )

    browser_cfg = BrowserConfig(
        headless=True, 
        verbose=False, 
        channel="chrome",
        # user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        # headers={"Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7"},
        # enable_stealth=True,
        # text_mode=True,
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)

    debug_payload = {
        "html": result.html,
        "cleaned_html": result.cleaned_html,
        "fit_html": result.fit_html,
        "markdown": {
            "raw_markdown": result.markdown.raw_markdown if result.markdown else None,
            "fit_markdown": result.markdown.fit_markdown if result.markdown else None,
            "markdown_with_citations": result.markdown.markdown_with_citations if result.markdown else None,
            "references_markdown": result.markdown.references_markdown if result.markdown else None,
        },
        "llm": llm_strategy.debug_payload,
    }

    if not result.success:
        return {
            "success": False,
            "data": None,
            "error": result.error_message,
            "debug": debug_payload,
        }

    try:
        data = json.loads(result.extracted_content)
    except json.JSONDecodeError:
        data = {"raw": result.extracted_content}

    return {"success": True, "data": data, "error": None, "debug": debug_payload}
