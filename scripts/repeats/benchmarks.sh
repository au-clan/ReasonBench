#!/bin/bash

benchmarks=(
  "game24"
  "hle"
  "hotpotqa"
  "humaneval"
  "scibench"
  "sonnetwriting"
)
method="io"
split="test"
repeats=10

provider="openai"
api_key="OPENAI_API_KEY"
model="gpt-4.1-nano"

source scripts/configs/$benchmark.env

for benchmark in "${benchmarks[@]}"; do
  echo "Running benchmark=$benchmark with method=$method"

  # Reset MAX_COMPLETION_TOKENS to default from env
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

  echo "Finished benchmark=$benchmark"
  echo "-----------------------------------"
done
