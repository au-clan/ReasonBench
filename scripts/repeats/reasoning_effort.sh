#!/bin/bash

# Choose multiple benchmarks + models
benchmarks=(
    "game24"
    "hle"
    "hotpotqa"
    "humaneval"
    "scibench"
    "sonnetwriting"
)

models=(
  "gemini-3-flash-preview"
)

# Loop over reasoning effort levels
reasoning_efforts=(
  "minimal"
  "low"
  "medium"
  "high"
)

method="io"
split="test"
repeats=10

provider="gemini"
api_key="GEMINI_API_KEY"

for benchmark in "${benchmarks[@]}"; do
  echo "==============================="
  echo "Loading config for benchmark=$benchmark"
  echo "==============================="

  # Load benchmark-specific config (TEMPERATURE, TOP_P, STOP, MAX_COMPLETION_TOKENS, etc.)
  source "scripts/configs/${benchmark}.env"

  for model in "${models[@]}"; do
    for reasoning_effort in "${reasoning_efforts[@]}"; do
      echo "Running benchmark=$benchmark with model=$model reasoning_effort=$reasoning_effort"

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
          --reasoning_effort "$reasoning_effort" \
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

      echo "Finished benchmark=$benchmark model=$model reasoning_effort=$reasoning_effort"
      echo "-----------------------------------"
    done
  done
done
