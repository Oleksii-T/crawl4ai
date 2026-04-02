import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from crawl4ai.extraction_strategy import (
    LLMExtractionStrategy,
    PROMPT_EXTRACT_BLOCKS,
    PROMPT_EXTRACT_BLOCKS_WITH_INSTRUCTION,
    PROMPT_EXTRACT_INFERRED_SCHEMA,
    PROMPT_EXTRACT_SCHEMA_WITH_INSTRUCTION,
    TokenUsage,
    extract_xml_data,
    split_and_parse_json_objects,
)
from crawl4ai.async_configs import LLMConfig
from crawl4ai.config import (
    CHUNK_TOKEN_THRESHOLD,
    DEFAULT_PROVIDER,
    DEFAULT_PROVIDER_API_KEY,
    OVERLAP_RATE,
    WORD_TOKEN_RATE,
)
from crawl4ai.utils import (
    aperform_completion_with_backoff,
    escape_json_string,
    sanitize_html,
    sanitize_input_encode,
)


class DebugLLMExtractionStrategy(LLMExtractionStrategy):
    """Capture the exact prompts and chunk payloads sent to the upstream LLM."""

    def __init__(
        self,
        llm_config: LLMConfig = None,
        instruction: str = None,
        schema: Dict = None,
        extraction_type: str = "block",
        chunk_token_threshold=CHUNK_TOKEN_THRESHOLD,
        overlap_rate=OVERLAP_RATE,
        word_token_rate=WORD_TOKEN_RATE,
        apply_chunking: bool = True,
        input_format: str = "markdown",
        force_json_response: bool = False,
        verbose: bool = False,
        provider: str = DEFAULT_PROVIDER,
        api_token: Optional[str] = None,
        base_url: str = None,
        api_base: str = None,
        **kwargs,
    ):
        super().__init__(
            llm_config=llm_config,
            instruction=instruction,
            schema=schema,
            extraction_type=extraction_type,
            chunk_token_threshold=chunk_token_threshold,
            overlap_rate=overlap_rate,
            word_token_rate=word_token_rate,
            apply_chunking=apply_chunking,
            input_format=input_format,
            force_json_response=force_json_response,
            verbose=verbose,
            provider=provider,
            api_token=api_token,
            base_url=base_url,
            api_base=api_base,
            **kwargs,
        )
        self.debug_payload: Dict[str, Any] = {
            "merged_sections": [],
            "requests": [],
        }

    def _build_prompt(self, url: str, html: str) -> str:
        variable_values = {
            "URL": url,
            "HTML": escape_json_string(sanitize_html(html)),
        }

        prompt_with_variables = PROMPT_EXTRACT_BLOCKS
        if self.instruction:
            variable_values["REQUEST"] = self.instruction
            prompt_with_variables = PROMPT_EXTRACT_BLOCKS_WITH_INSTRUCTION

        if self.extract_type == "schema" and self.schema:
            variable_values["SCHEMA"] = json.dumps(self.schema, indent=2)
            prompt_with_variables = PROMPT_EXTRACT_SCHEMA_WITH_INSTRUCTION

        if self.extract_type == "schema" and not self.schema:
            prompt_with_variables = PROMPT_EXTRACT_INFERRED_SCHEMA

        for variable, value in variable_values.items():
            prompt_with_variables = prompt_with_variables.replace(
                "{" + variable + "}", value
            )

        return prompt_with_variables

    async def aextract(self, url: str, ix: int, html: str) -> List[Dict[str, Any]]:
        if self.verbose:
            print(f"[LOG] Call LLM for {url} - block index: {ix}")

        prompt_with_variables = self._build_prompt(url, html)
        request_record: Dict[str, Any] = {
            "index": ix,
            "sanitized_section": html,
            "prompt": prompt_with_variables,
        }

        try:
            response = await aperform_completion_with_backoff(
                self.llm_config.provider,
                prompt_with_variables,
                self.llm_config.api_token,
                base_url=self.llm_config.base_url,
                json_response=self.force_json_response,
                extra_args=self.extra_args,
                base_delay=self.llm_config.backoff_base_delay,
                max_attempts=self.llm_config.backoff_max_attempts,
                exponential_factor=self.llm_config.backoff_exponential_factor,
            )
            usage = TokenUsage(
                completion_tokens=response.usage.completion_tokens,
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
                completion_tokens_details=response.usage.completion_tokens_details.__dict__
                if response.usage.completion_tokens_details
                else {},
                prompt_tokens_details=response.usage.prompt_tokens_details.__dict__
                if response.usage.prompt_tokens_details
                else {},
            )
            self.usages.append(usage)
            self.total_usage.completion_tokens += usage.completion_tokens
            self.total_usage.prompt_tokens += usage.prompt_tokens
            self.total_usage.total_tokens += usage.total_tokens

            content = response.choices[0].message.content
            request_record["response"] = content
            request_record["usage"] = {
                "completion_tokens": usage.completion_tokens,
                "prompt_tokens": usage.prompt_tokens,
                "total_tokens": usage.total_tokens,
            }

            try:
                if self.force_json_response:
                    blocks = json.loads(content)
                    if isinstance(blocks, dict):
                        if len(blocks) == 1 and isinstance(list(blocks.values())[0], list):
                            blocks = list(blocks.values())[0]
                        else:
                            blocks = [blocks]
                else:
                    blocks = extract_xml_data(["blocks"], content)["blocks"]
                    blocks = json.loads(blocks)

                for block in blocks:
                    block["error"] = False
            except Exception:
                parsed, unparsed = split_and_parse_json_objects(content)
                blocks = parsed
                if unparsed:
                    blocks.append(
                        {
                            "index": 0,
                            "error": True,
                            "tags": ["error"],
                            "content": unparsed,
                        }
                    )

            request_record["parsed_blocks"] = blocks
            self.debug_payload["requests"].append(request_record)

            if self.verbose:
                print(
                    "[LOG] Extracted",
                    len(blocks),
                    "blocks from URL:",
                    url,
                    "block index:",
                    ix,
                )
            return blocks
        except Exception as exc:
            request_record["error"] = str(exc)
            self.debug_payload["requests"].append(request_record)
            if self.verbose:
                print(f"[LOG] Error in LLM extraction: {exc}")
            return [
                {
                    "index": ix,
                    "error": True,
                    "tags": ["error"],
                    "content": str(exc),
                }
            ]

    async def arun(self, url: str, sections: List[str]) -> List[Dict[str, Any]]:
        merged_sections = self._merge(
            sections,
            self.chunk_token_threshold,
            overlap=int(self.chunk_token_threshold * self.overlap_rate),
        )
        self.debug_payload["merged_sections"] = merged_sections

        tasks = [
            self.aextract(url, ix, sanitize_input_encode(section))
            for ix, section in enumerate(merged_sections)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        extracted_content: List[Dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                if self.verbose:
                    print(f"Error in async extraction: {result}")
                extracted_content.append(
                    {
                        "index": 0,
                        "error": True,
                        "tags": ["error"],
                        "content": str(result),
                    }
                )
            else:
                extracted_content.extend(result)

        return extracted_content
