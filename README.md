# ReasonBENCH: Benchmarking the (In)Stability of LLM Reasoning

**ReasonBENCH** is a benchmark suite and open-source library for controlled multi-run evaluation of LLM reasoning. It measures both the quality and stability of reasoning strategies by running repeated independent trials and reporting variance-aware metrics — including confidence intervals, run deviation, and global noise — rather than relying on single-run averages.

> *Preliminary work. Under review by the International Conference on Machine Learning (ICML).*

<!-- Leaderboard: http://reasonbench.github.io -->

## Motivation

LLM reasoning is typically evaluated using single runs, masking how much performance can vary across repeated executions. This practice obscures both reliability and cost, and can lead to misleading comparisons between methods and models. ReasonBENCH addresses this by repeating every model-strategy-task configuration with 10 independent trials and reporting distributional metrics alongside averages.

Key findings from our evaluation:
- **Run-to-run variability is substantial** — often large enough to change model/method rankings relative to single-run averages
- **Quality and cost stability decouple** — the most accurate strategy is not necessarily the most stable, and vice versa
- **Model scaling improves both quality and stability** — larger models within a family yield tighter distributions
- **Prompt refinements improve quality but not stability** — clarifying prompts and parsers boosts accuracy without reducing run-to-run variance
- **Reasoning effort scales cost, not quality** — increasing test-time reasoning effort primarily raises cost with limited and statistically insignificant quality gains

## Reasoning Strategies

We implement 10 representative reasoning strategies using a standardized interface:

| Strategy | Type | Reference |
|----------|------|-----------|
| **IO** | Direct | — |
| **CoT** | Direct | Wei et al., 2022 |
| **CoT-SC** | Direct | Wang et al., 2023 |
| **ReAct** | Adaptive | Yao et al., 2023b |
| **Reflexion** | Adaptive | Shinn et al., 2023 |
| **ToT-BFS** | Structured | Yao et al., 2023a |
| **ToT-DFS** | Structured | Yao et al., 2023a |
| **GoT** | Structured | Besta et al., 2024 |
| **RAP** | Planning | Hao et al., 2023 |
| **FoA** | Evolutionary | Klein et al., 2025 |

## Benchmarks

6 tasks spanning diverse reasoning domains:

| Task | Domain | Metric | Size |
|------|--------|--------|------|
| **Game of 24** | Mathematical reasoning | Accuracy | 100 |
| **SciBench** | Scientific reasoning | Accuracy (exact match) | 109 |
| **HumanEval** | Code generation | pass@1 | 100 |
| **HotPotQA** | Multi-hop QA | Exact match | 100 |
| **Sonnet Writing** | Creative writing | Accuracy (rhyme + words) | 50 |
| **HLE** | General reasoning (Humanity's Last Exam) | Accuracy | 50 |

## Evaluated Models

10 contemporary reasoning models from 6 providers:

| Model | Provider |
|-------|----------|
| GPT-4.1 Nano, GPT-4.1 Mini | OpenAI |
| GPT-5 Nano, GPT-5 Mini | OpenAI |
| GPT-OSS 120B | Together AI |
| DeepSeek R1 | Together AI |
| Llama 4 Maverick | Together AI |
| Qwen3-235B Thinking | Together AI |
| Claude Haiku 4.5 | Anthropic |
| Gemini 3 Flash | Google |

## Setup

```bash
pip install -r requirements.txt
```

You also need [CacheSaver](https://github.com/au-clan/cachesaver) — a client-side inference optimization framework for efficient, affordable, and reproducible LLM inference:

```bash
pip install cachesaver
```

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="sk-..."
# and/or other provider keys
```

## Quick Start

The simplest way to run an experiment is via the shell script:

```bash
bash scripts/simple/simple.sh
```

Edit the variables at the top of `scripts/simple/simple.sh` to change the benchmark, method, model, and split.

For direct invocation:

```bash
python scripts/simple/simple.py \
    --benchmark game24 \
    --method tot_bfs \
    --split mini \
    --provider openai \
    --api_key OPENAI_API_KEY \
    --model gpt-4.1-nano \
    --temperature 1.0 \
    --max_completion_tokens 10000 \
    --top_p 1.0 \
    --batch_size 1 \
    --timeout 2.0 \
    --correctness 1 \
    --allow_batch_overflow 1 \
    --ns_ratio 0.0 \
    --value_cache
```

### Key arguments

| Argument | Description |
|----------|-------------|
| `--benchmark` | Task name: `game24`, `humaneval`, `hotpotqa`, `scibench`, `hle`, `sonnetwriting` |
| `--method` | Reasoning method: `io`, `cot`, `cot_sc`, `foa`, `tot_bfs`, `tot_dfs`, `got`, `react`, `rap` |
| `--split` | Dataset split: `train`, `validation`, `test`, `mini` |
| `--provider` | LLM provider: `openai`, `gemini`, `anthropic`, `groq`, `together` |
| `--model` | Model identifier (e.g., `gpt-4.1-nano`, `claude-haiku-4-5`) |
| `--ns_ratio` | Namespace ratio (0.0–1.0) for controlling parallel execution |

## Evaluation Metrics

For each model-strategy-task configuration, we report metrics along two dimensions:

**Quality:**
- **Average** — stratified bootstrap mean over runs; benchmarks treated as strata
- **Run Deviation** — typical run-to-run deviation from the strategy mean per benchmark
- **Noise (Global)** — variance of z-scored outcomes across all benchmarks
- **Noise (Run)** — average within-benchmark z-score variance

**Cost:**
- Same four metrics computed over token usage and wall-clock time, expressed in USD

## Configuration

Method hyperparameters are defined per task in YAML files under `scripts/configs/`:

```yaml
# scripts/configs/game24.yaml
tot_bfs:
  num_selections: 3
  num_steps: 4
  num_evaluations: 3

got:
  num_selections: 5
  num_steps: 4
  num_generate: 10
  num_evaluations: 3
  num_best: 2
```

Decoding parameters (temperature, top_p, max tokens) are sourced from `scripts/configs/<task>.env`.

## Architecture

ReasonBENCH is organized around four core abstractions:

- **Method** — specifies the reasoning strategy independently of the model or task. Integrates agents, the environment, and the model, and exposes a standard `solve()` interface.
- **Environment** — formalizes task-specific dynamics: state transitions, action validation, terminal conditions, and evaluation.
- **Agent** — defines the interface between methods, models, and states. Agents construct prompts, issue queries, and parse responses into actions.
- **Model** — uniform interface for LLM providers, supporting async execution and integrated with CacheSaver for response caching and deduplication.

```
src/
├── models/          # LLM provider adapters (OpenAI, Anthropic, Groq, Together, Gemini)
├── methods/         # Reasoning strategy implementations
├── tasks/           # Task definitions (state, environment, agents, prompts)
│   ├── game24/
│   ├── humaneval/
│   ├── hotpotqa/
│   └── ...
├── __init__.py      # Factory registrations
├── typedefs.py      # Core ABCs and type definitions
└── utils.py         # Logging and utility functions

scripts/
├── simple/          # Single-run experiment scripts
├── repeats/         # Batch/repeated experiment scripts
├── cached/          # Cached inference scripts
└── configs/         # YAML and .env configuration files

datasets/            # Gzip-compressed task datasets
tests/               # Pytest test suite
```

## Tests

```bash
pytest                                       # run all tests
pytest tests/got/test_game24.py              # single file
pytest tests/got/test_game24.py -k "test_x"  # single test
```

Tests use async fixtures and require valid API keys (Groq/OpenAI) for the mock LLM clients.

## Citation

```bibtex
@article{potamitis2025reasonbench,
  title={ReasonBENCH: Benchmarking the (In) Stability of LLM Reasoning},
  author={Potamitis, Nearchos and Klein, Lars and Arora, Akhil},
  journal={arXiv preprint arXiv:2512.07795},
  year={2025}
}
```