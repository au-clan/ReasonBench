import os
import asyncio    
from typing import List, Any
from dataclasses import dataclass

from cachesaver.typedefs import Request, Batch, Response

from ..typedefs import Model

class OnlineLLM(Model):
    def __init__(self, provider: str, max_n: int = 128, api_key: str=None, reasoning_effort=None):
        self.client = client_init(provider, api_key)
        self.max_n = max_n if provider not in ["groq", "anthropic"] else 1
        self.reasoning_effort = reasoning_effort

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
                    completion = await chat_completion(self.client, prompts, request, current_n, self.reasoning_effort)
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Sleeping for: {max(sleep, 90)} seconds")
                    await asyncio.sleep(max(sleep, 90))
                    sleep *= 2

            input_tokens, completion_tokens = count_tokens(completion)
            
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
    # OpenAI - GPT
    if provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv(api_key))
        return client
    
    # Google - Gemini
    elif provider == "gemini":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key = os.getenv(api_key),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        return client
    
    # Anthropic - Claude
    elif provider == "anthropic":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key = os.getenv(api_key),
            base_url="https://api.anthropic.com/v1/"
        )
        return client
    
    # Together - HF
    elif provider == "together":
        from together import AsyncTogether
        client = AsyncTogether(api_key=os.getenv(api_key))
        return client
    
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
    

async def chat_completion(client, prompts, request, current_n, reasoning_effort: str=None):
    model_name = request.kwargs["model"]
    if model_name.startswith(("gemini", "gemma")):
        completion = await google_chat_completion(client, prompts, request, current_n, reasoning_effort)
    elif model_name.startswith(("claude")):
        completion = await anthropic_chat_completion(client, prompts, request, current_n, reasoning_effort)
    else:
        completion = await openai_chat_completion(client, prompts, request, current_n, reasoning_effort)
    return completion


async def anthropic_chat_completion(client, prompts, request, current_n, reasoning_effort: str=None):
    if reasoning_effort:
        completion = await client.chat.completions.create(
            messages=prompts,
            model=request.kwargs["model"],
            n=1, # Anthropic restriction
            max_completion_tokens=request.kwargs.get("max_completion_tokens") or None,
            temperature=request.kwargs.get("temperature", None) or 1,
            #stop=request.kwargs.get("stop", None) or None,
            #top_p=request.kwargs.get("top_p", None) or 1,
            #seed=request.kwargs.get("seed", None) or None,
            #logprobs=request.kwargs.get("logprobs", None) or False,
            #top_logprobs=request.kwargs.get("top_logprobs", None) or None,
            extra_body = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": reasoning_effort
                }
            }
        )
    else:
        completion = await client.chat.completions.create(
            messages=prompts,
            model=request.kwargs["model"],
            n=1, # Anthropic restriction
            max_completion_tokens=request.kwargs.get("max_completion_tokens") or None,
            temperature=request.kwargs.get("temperature", None) or 1,
            #stop=request.kwargs.get("stop", None) or None,
            #top_p=request.kwargs.get("top_p", None) or 1,
            #seed=request.kwargs.get("seed", None) or None,
            #logprobs=request.kwargs.get("logprobs", None) or False,
            #top_logprobs=request.kwargs.get("top_logprobs", None) or None
        )
    return completion

async def google_chat_completion(client, prompts, request, current_n, reasoning_effort: str=None):
    if reasoning_effort:
        completion = await client.chat.completions.create(
            messages=prompts,
            model=request.kwargs["model"],
            n=current_n,
            max_completion_tokens=request.kwargs.get("max_completion_tokens") or None,
            temperature=request.kwargs.get("temperature", None) or 1,
            #stop=request.kwargs.get("stop", None) or [""],
            top_p=request.kwargs.get("top_p", None) or 1,
            #seed=request.kwargs.get("seed", None) or None,
            #logprobs=request.kwargs.get("logprobs", None) or False,
            #top_logprobs=request.kwargs.get("top_logprobs", None) or None,
            extra_body = {
                "extra_body": {
                    "google": {
                        "thinking_config": {
                            "thinking_level": reasoning_effort,
                            "include_thoughts": False
                        }
                    }
                }
            }
        )
    else:
        completion = await client.chat.completions.create(
            messages=prompts,
            model=request.kwargs["model"],
            n=current_n,
            max_completion_tokens=request.kwargs.get("max_completion_tokens") or None,
            temperature=request.kwargs.get("temperature", None) or 1,
            #stop=request.kwargs.get("stop", None) or [""],
            top_p=request.kwargs.get("top_p", None) or 1,
            #seed=request.kwargs.get("seed", None) or None,
            #logprobs=request.kwargs.get("logprobs", None) or False,
            #top_logprobs=request.kwargs.get("top_logprobs", None) or None
        )
    return completion

    

async def openai_chat_completion(client, prompts, request, current_n, reasoning_effort: str=None):
    if reasoning_effort:
        completion = await client.chat.completions.create(
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
            reasoning_effort=reasoning_effort
        )
    
    else:
        completion = await client.chat.completions.create(
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
    return completion

def count_tokens(completion):
    input_tokens = completion.usage.prompt_tokens
    if completion.model.startswith(("gemini", "gemma")):
        completion_tokens = completion.usage.total_tokens - completion.usage.prompt_tokens 
    else:
        completion_tokens = completion.usage.completion_tokens
    return input_tokens, completion_tokens
