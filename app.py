import argparse
import asyncio
import json
import os

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMConfig,
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from dotenv import load_dotenv

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run crawl4ai with custom params.")
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--instructions", type=str, required=True)
    parser.add_argument(
        "--schema",
        type=str,
        required=True,
        help="JSON schema as a string, or shorthand like {\"title\":\"string\"}.",
    )
    return parser.parse_args()


def resolve_schema(schema_str: str) -> dict:
    if not schema_str:
        raise SystemExit("Missing --schema JSON string.")
    try:
        parsed = json.loads(schema_str)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --schema JSON: {exc}")
    if isinstance(parsed, dict):
        if "type" in parsed or "properties" in parsed:
            return parsed
        if parsed and all(isinstance(value, str) for value in parsed.values()):
            return {
                "title": "CustomSchema",
                "type": "object",
                "properties": {key: {"type": value} for key, value in parsed.items()},
                "required": list(parsed.keys()),
            }
    raise SystemExit(
        "Invalid --schema JSON: expected a JSON schema object or shorthand mapping like "
        '{"title":"string","url":"string"}.'
    )


async def main(args: argparse.Namespace):
    load_dotenv()
    os.environ.setdefault("CRAWL4_AI_BASE_DIRECTORY", os.getcwd())
    os.makedirs(os.path.join(os.getcwd(), ".crawl4ai"), exist_ok=True)
    # os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", os.path.join(os.getcwd(), ".playwright"))
    # os.makedirs(os.environ["PLAYWRIGHT_BROWSERS_PATH"], exist_ok=True)

    llm_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="deepseek/deepseek-chat",
            api_token=os.getenv("DEEPSEEK_API"),
            temperature=0.0,
            max_tokens=800,
        ),
        schema=resolve_schema(args.schema),
        extraction_type="schema",
        instruction=args.instructions,
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
    )

    browser_cfg = BrowserConfig(headless=True, verbose=False, channel="chrome")

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=args.url, config=crawl_config)

        if result.success:
            data = json.loads(result.extracted_content)

            print("Extracted items:", data)

            # llm_strategy.show_usage()
        else:
            print("Error:", result.error_message)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
