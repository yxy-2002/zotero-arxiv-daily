from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import ast
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json

RawPaperItem = TypeVar('RawPaperItem')


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _generate_tldr_with_llm(self, openai_client: OpenAI, llm_params: dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"

        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"

        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]
        prompt = enc.decode(prompt_tokens)

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant specialized in summarizing scientific papers. "
                        "Always answer in Simplified Chinese. "
                        "Be accurate, concise, and avoid hype. "
                        "Return exactly one sentence for the TL;DR."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        tldr = response.choices[0].message.content
        return tldr

    def generate_tldr(self, openai_client: OpenAI, llm_params: dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client, llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _parse_affiliations_response(self, response_text: str) -> list[str]:
        match = re.search(r'\[.*?\]', response_text, flags=re.DOTALL)
        if match is None:
            logger.warning(f"No affiliation list found in LLM response for {self.url}: {response_text}")
            return []

        list_text = match.group(0)

        try:
            affiliations = json.loads(list_text)
        except json.JSONDecodeError:
            try:
                affiliations = ast.literal_eval(list_text)
            except (SyntaxError, ValueError) as e:
                logger.warning(f"Failed to parse affiliation list for {self.url}: {e}")
                return []

        if not isinstance(affiliations, list):
            logger.warning(f"Affiliations response is not a list for {self.url}: {affiliations}")
            return []

        cleaned_affiliations = []
        seen = set()

        for affiliation in affiliations:
            if affiliation is None:
                continue

            affiliation = str(affiliation).strip()
            if not affiliation or affiliation in seen:
                continue

            seen.add(affiliation)
            cleaned_affiliations.append(affiliation)

        return cleaned_affiliations

    def _generate_affiliations_with_llm(
        self,
        openai_client: OpenAI,
        llm_params: dict
    ) -> list[str]:
        if self.full_text is None:
            logger.warning(f"No full text is provided for affiliation extraction: {self.url}")
            return []

        prompt = (
            "Given the beginning of a paper, extract the affiliations of the authors "
            "in a python list format, which is sorted by the author order. "
            "If there is no affiliation found, return an empty list '[]':\n\n"
            f"{self.full_text}"
        )

        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:2000]
        prompt = enc.decode(prompt_tokens)

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant specialized in extracting author affiliations from academic papers. "
                        "Always answer in valid Python list format. "
                        "Return only the final list and nothing else."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )

        response_text = response.choices[0].message.content or ""
        return self._parse_affiliations_response(response_text)

    def generate_affiliations(
        self,
        openai_client: OpenAI,
        llm_params: dict
    ) -> list[str]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client, llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = []
            return []


@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
