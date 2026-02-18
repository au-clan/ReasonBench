import os
import time
import asyncio
import argparse

from diskcache import Cache
from omegaconf import OmegaConf

import logging
logger = logging.getLogger(__name__)

from cachesaver.pipelines import OnlineAPI

import sys
sys.path.append(os.getcwd())

from src import BenchmarkFactory, EnvironmentFactory, MethodFactory
from src.tasks import *
from src.methods import *
from src.models import OnlineLLM, API
from src.typedefs import DecodingParameters

from src.utils import initial_logging, final_logging


async def run(args, cache_path):
    
    # Cache directory
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = Cache(cache_path)

    # Model
    model = OnlineLLM(provider=args.provider, api_key=args.api_key, reasoning_effort=args.reasoning_effort)

    # Pipeline
    pipeline = OnlineAPI(
        model=model,
        cache=cache,
        batch_size=args.batch_size,
        timeout=args.timeout,
        allow_batch_overflow=args.allow_batch_overflow,
        correctness=bool(args.correctness)
    )

    # API
    api = API(
        pipeline=pipeline,
        model=args.model,
        log_path=f"logs/raw_calls/simple/{args.model}/{args.benchmark}/{args.method}_{args.split}.log"
    )

    # Decoding Parameters
    params = DecodingParameters(
        temperature=1.0 if args.model.startswith("gpt-5") else args.temperature, # GPT-5* limited to temperature 1.0
        max_completion_tokens=args.max_completion_tokens,
        top_p=args.top_p,
        stop=args.stop,
        logprobs=args.logprobs
    )

    # Config for the framework hyperparameters
    config = OmegaConf.load(f"scripts/configs/{args.benchmark}.yaml")[args.method]

    # Environment
    environment = EnvironmentFactory.get(args.benchmark)

    # Method
    method = MethodFactory.get(
        method=args.method,
        benchmark=args.benchmark,
        params=params,
        model=api,
        env=environment,
        config=config)
    

    # Benchmark
    benchmark = BenchmarkFactory.get(args.benchmark, split=args.split, max_len=50)

    for i in range(int(args.repeats)):
        print(f"Repeat {i+1}/{args.repeats}")

        # Clean the API costs and update the log_path (not the cache though)
        api.clean()
        if args.reasoning_effort:
            calls_path = f"logs/raw_calls/simple/{args.model}_{args.reasoning_effort}/{args.benchmark}/{args.method}_{args.split}_{i}.log"
        else:
            calls_path = f"logs/raw_calls/simple/{args.model}/{args.benchmark}/{args.method}_{args.split}_{i}.log"
        api.update_log_path(
            log_path=calls_path
        )
        
        # Initial logging
        if args.reasoning_effort:
            log_path = f"logs/repeats/{args.model}_{args.reasoning_effort}/{args.benchmark}/{args.method}_{args.split}_{i}.log"
        else:
            log_path = f"logs/repeats/{args.model}/{args.benchmark}/{args.method}_{args.split}_{i}.log"
        initial_logging(logger, args, log_path)

        # Start timing
        start = time.perf_counter()

        # Run the method 
        durations, results = await method.benchmark(
            benchmark=benchmark,
            ns_ratio=args.ns_ratio,
            **({"value_cache": True} if bool(args.value_cache) else {})
        )

        # End timing
        end = time.perf_counter()
        clocktime = end - start

        # Final logging
        evaluations = [sorted([environment.evaluate(state) for state in r], key=lambda x: x[1]) for r in results]
        final_logging(logger, api, clocktime, durations, evaluations)

        



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Solve tasks with different methods.")

    parser.add_argument("--repeats", type=int)
    
    parser.add_argument("--method", type=str)
    parser.add_argument("--benchmark", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--value_cache", action="store_true")

    parser.add_argument("--provider", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--reasoning_effort", type=str, default=None)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max_completion_tokens", type=int)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--stop", type=str, nargs="+")
    parser.add_argument("--logprobs", action="store_true")

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--allow_batch_overflow", type=int)
    parser.add_argument("--ns_ratio", type=float)
    parser.add_argument("--correctness", type=int)
    args = parser.parse_args()

    if args.ns_ratio > 1.0 or args.ns_ratio < 0.0:
        raise ValueError("ns_ratio must be between 0.0 and 1.0")
    
    cache_path = "caches/developpi"
    if args.reasoning_effort:
        cache_path = f"{cache_path}_{args.reasoning_effort}"

    asyncio.run(run(args, cache_path=cache_path))