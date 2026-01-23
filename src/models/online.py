import os
import asyncio    
from typing import List, Any
from dataclasses import dataclass

from cachesaver.typedefs import Request, Batch, Response

from ..typedefs import Model

class OnlineLLM(Model):
    def __init__(self, provider: str, max_n: int = 128, api_key: str=None):
        self.client = client_init(provider, api_key)
        self.max_n = max_n

    async def request(self, request: Request) -> Response:
        total_n = request.n
        results = []
        input_tokens = 0
        completion_tokens = 0
        sleep = 1

        prompts = (
            [{"role": "user", "content": request.kwargs["prompt"]}]
            if isinstance(request.kwargs["prompt"], str)
            else request.kwargs["prompt"]
        )

        while total_n > 0:
            current_n = min(total_n, self.max_n)
            total_n -= current_n

            while True:
                try:
                    completion = await self.client.chat.completions.create(
                        messages=prompts,
                        model=request.kwargs["model"],
                        n=current_n,
                        max_completion_tokens=request.kwargs.get("max_completion_tokens") or None,
                        temperature=request.kwargs.get("temperature", None) or 1,
                        stop=request.kwargs.get("stop", None) or None,
                        top_p=request.kwargs.get("top_p", None) or 1,
                        seed=request.kwargs.get("seed", None) or None,
                        logprobs=request.kwargs.get("logprobs", None) or False,
                        top_logprobs=request.kwargs.get("top_logprobs", None) or None,
                    )
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Sleeping for: {max(sleep, 90)} seconds")
                    await asyncio.sleep(max(sleep, 90))
                    sleep *= 2

            input_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            if getattr(completion.usage, 'prompt_tokens_details', None):
                try:
                    cached_tokens = completion.usage.prompt_tokens_details.cached_tokens
                    
                except Exception as e:
                    print(f"Could not access cached tokens: {e}")
                    pass
            else:
                cached_tokens = 0

            results.extend(
                (choice.message.content, input_tokens, completion_tokens / current_n, cached_tokens)
                for choice in completion.choices
            )

        return Response(data=results)

    
    async def batch_request(self, batch: Batch) -> List[Response]:
        requests = [self.request(request) for request in batch.requests]
        completions = await asyncio.gather(*requests)
        return completions
    
def client_init(provider: str, api_key: str) -> Any:
    if provider == "openai":
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=os.getenv(api_key))
    elif provider == "together":
        from together import AsyncTogether
        return AsyncTogether(api_key=os.getenv(api_key))
    else:
        raise ValueError(f"Unknown provider: {provider}")