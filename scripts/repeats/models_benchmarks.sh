#!/bin/bash

# Choose multiple benchmarks + models
benchmarks=(
    "hle"
    "game24"
    "hotpotqa"
    "humaneval"
    "scibench"
    "sonnetwriting"
)
models=(
  "claude-haiku-4-5-20251001"
)

method="io"
split="test"
repeats=10

provider="anthropic"
api_key="ANTHROPIC_API_KEY"

for benchmark in "${benchmarks[@]}"; do
  echo "==============================="
  echo "Loading config for benchmark=$benchmark"
  echo "==============================="

  # Load benchmark-specific config (TEMPERATURE, TOP_P, STOP, MAX_COMPLETION_TOKENS, etc.)
  source "scripts/configs/${benchmark}.env"

  for model in "${models[@]}"; do
    echo "Running benchmark=$benchmark with model=$model"

    # Reset MAX_COMPLETION_TOKENS to default from env for EACH run
    MAX_COMPLETION_TOKENS_DEFAULT="$MAX_COMPLETION_TOKENS"

    # Override for specific methods
    if [[ "$method" == "io" || "$method" == "cot" ]]; then
      MAX_COMPLETION_TOKENS=10000
    else
      MAX_COMPLETION_TOKENS="$MAX_COMPLETION_TOKENS_DEFAULT"
    fi

    python scripts/repeats/repeats.py \
        --benchmark "$benchmark" \
        --method "$method" \
        --model "$model" \
        --batch_size 1 \
        --timeout 2.0 \
        --temperature "$TEMPERATURE" \
        --max_completion_tokens "$MAX_COMPLETION_TOKENS" \
        --top_p "$TOP_P" \
        --dataset_path "datasets/dataset_${benchmark}.csv.gz" \
        --split "$split" \
        --correctness 1 \
        --allow_batch_overflow 1 \
        --ns_ratio 0.0 \
        --provider "$provider" \
        --api_key "$api_key" \
        ${STOP:+--stop "$STOP"} \
        --value_cache \
        --repeats "$repeats"

    echo "Finished benchmark=$benchmark model=$model"
    echo "-----------------------------------"
  done
done
