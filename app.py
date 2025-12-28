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
from pydantic import BaseModel, Field

URL_TO_SCRAPE = "https://web.lmarena.ai/leaderboard"

INSTRUCTION_TO_LLM = "Extract all rows from the main table as objects with 'rank', 'model', 'arena score', '95% CI', 'Votes', 'Organization', 'License' from the content."


class ArenaRow(BaseModel):
    rank: str
    model: str
    arena_score: str = Field(alias="arena score")
    ci_95: str = Field(alias="95% CI")
    votes: str = Field(alias="Votes")
    organization: str = Field(alias="Organization")
    license: str = Field(alias="License")


async def main():
    load_dotenv()
    os.environ.setdefault("CRAWL4_AI_BASE_DIRECTORY", os.getcwd())
    os.makedirs(os.path.join(os.getcwd(), ".crawl4ai"), exist_ok=True)
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", os.path.join(os.getcwd(), ".playwright"))
    os.makedirs(os.environ["PLAYWRIGHT_BROWSERS_PATH"], exist_ok=True)

    llm_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="deepseek/deepseek-chat",
            api_token=os.getenv("DEEPSEEK_API"),
            temperature=0.0,
            max_tokens=800,
        ),
        schema=ArenaRow.model_json_schema(),
        extraction_type="schema",
        instruction=INSTRUCTION_TO_LLM,
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",
    )

    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS,
        process_iframes=False,
        remove_overlay_elements=True,
        exclude_external_links=True,
    )

    browser_cfg = BrowserConfig(headless=True, verbose=True, channel="chrome")

    async with AsyncWebCrawler(config=browser_cfg) as crawler:

        result = await crawler.arun(url=URL_TO_SCRAPE, config=crawl_config)

        if result.success:
            data = json.loads(result.extracted_content)

            print("Extracted items:", data)

            llm_strategy.show_usage()
        else:
            print("Error:", result.error_message)


if __name__ == "__main__":
    asyncio.run(main())
